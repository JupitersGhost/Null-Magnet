#![recursion_limit = "256"]

//! NullMagnet Live Core v1.0 - Jupiter Labs
//!
//! NIST SP 800-90B Aligned Entropy Harvesting with PQC Key Generation
//!
//! Architecture:
//!   lib.rs        - Engine, state, Tauri command bridge, mixer thread
//!   entropy.rs    - NIST health tests, entropy estimators, extraction pool
//!   harvesters.rs - All harvester threads (TRNG, audio, system, mouse, video,
//!                   GPU CUDA, GPU OpenCL, guitar UDP, WiFi, USB serial,
//!                   P2P server, headscale forwarder)
//!
//! NIST Compliance:
//! - Repetition Count Test (RCT) cutoff=31 (alpha=2^-30, H=1)
//! - Adaptive Proportion Test (APT) W=512, C=325
//! - Startup sample discard (4096 samples)
//! - SHA-256 vetted conditioning function
//! - Conservative entropy crediting (0.85 factor)
//! - Min-entropy estimation (Most Common Value + Collision)
//!
//! Live Concert Features:
//! - Guitar ESP32 UDP entropy (Spectra, Neptonius, Thalyn)
//! - Headscale forwarding to Aoi Midori (100.64.0.15)
//! - Live mode pulsing indicator
//! - Per-GPU independent threads (CUDA + OpenCL)

// ============================================================================
// MODULE DECLARATIONS
// ============================================================================

pub mod entropy;
pub mod harvesters;

// ============================================================================
// IMPORTS
// ============================================================================

use std::sync::{Arc, atomic::{AtomicBool, Ordering}};
use parking_lot::Mutex;
use crossbeam_channel::{bounded, Sender, Receiver};
use std::thread;
use std::time::Duration;
use std::fs;
use std::collections::{VecDeque, HashMap};
use sha2::{Sha256, Digest as Sha2Digest};
use sha3::Sha3_256;
use hmac::{Hmac, Mac};
use pqcrypto_kyber::kyber512;
use pqcrypto_falcon::falcon512;
use pqcrypto_traits::sign::{SecretKey as SignSecretKey, DetachedSignature};
use pqcrypto_traits::kem::{PublicKey as KemPublicKey, SecretKey as KemSecretKey};
use zeroize::Zeroize;
use pqcrypto_traits::sign::PublicKey as SignPublicKey;

// Re-export from entropy module
use entropy::{
    NistHealthTester, EntropyExtractionPool,
    EXTRACTION_POOL_SIZE, POOL_SIZE, HISTORY_LEN,
    NIST_RCT_CUTOFF, NIST_APT_WINDOW, NIST_APT_CUTOFF,
    NIST_CONDITIONING_FACTOR, STARTUP_DISCARD_SAMPLES,
    MIN_ENTROPY_FOR_MINT, AUTO_MINT_THRESHOLD,
    shannon_entropy, conservative_min_entropy, credit_entropy,
    get_timestamp, get_timestamp_nanos,
};

pub type HmacSha256 = Hmac<Sha256>;

// ============================================================================
// DATA STRUCTURES
// ============================================================================

#[derive(Clone, Default)]
pub struct SourceMetrics {
    pub raw_shannon: f64,
    pub min_entropy: f64,
    pub samples: u64,
    pub avg_raw_entropy: f64,
    pub total_bits_contributed: f64,
    pub health_state: String,
}

#[derive(Clone)]
pub struct P2PConfig {
    pub active: bool,
    pub listen_port: u16,
    pub peers: Vec<String>,
    pub received_count: u64,
    pub hmac_key: Option<Vec<u8>>,  // 32-byte HMAC key for authentication
}

impl Default for P2PConfig {
    fn default() -> Self {
        Self {
            active: false,
            listen_port: 9000,
            peers: Vec::new(),
            received_count: 0,
            hmac_key: None,
        }
    }
}

/// Per-guitar ESP32 tracking state
#[derive(Clone, Debug)]
pub struct GuitarState {
    pub name: String,
    pub data_port: u16,
    pub ctrl_port: u16,
    pub enabled: bool,
    pub packets_received: u64,
    pub bytes_received: u64,
}

/// Headscale forwarding configuration
#[derive(Clone)]
pub struct HeadscaleConfig {
    pub enabled: bool,
    pub target_ip: String,
    pub target_port: u16,
    pub forwarded_count: u64,
    pub last_forward_ts: u64,
}

impl Default for HeadscaleConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            target_ip: "100.64.0.15".to_string(),  // Aoi Midori
            target_port: 8100,
            forwarded_count: 0,
            last_forward_ts: 0,
        }
    }
}

/// Harvester enable/disable states
/// GPU CUDA and OpenCL are tracked independently (separate threads)
#[derive(Clone)]
pub struct HarvesterStates {
    pub trng: bool,
    pub audio: bool,
    pub system: bool,
    pub mouse: bool,
    pub video: bool,
    pub gpu_cuda: bool,     // NVIDIA CUDA (independent thread)
    pub gpu_ocl: bool,      // AMD/Intel OpenCL (independent thread)
    pub wifi: bool,         // WiFi noise harvester
    pub usb_serial: bool,   // USB serial device harvester
    // Guitar states tracked in SharedState.guitar_states
}

impl Default for HarvesterStates {
    fn default() -> Self {
        Self {
            trng: false,
            audio: false,
            system: false,
            mouse: false,
            video: false,
            gpu_cuda: false,
            gpu_ocl: false,
            wifi: false,
            usb_serial: false,
        }
    }
}

/// Device manager settings (runtime adjustable)
#[derive(Clone)]
pub struct DeviceConfig {
    pub audio_device_index: Option<usize>,
    pub audio_gain: f64,
    pub camera_device_index: usize,
    pub usb_serial_port: String,
    pub usb_serial_baud: u32,
    pub wifi_interface: String,
}

impl Default for DeviceConfig {
    fn default() -> Self {
        Self {
            audio_device_index: None,
            audio_gain: 1.0,
            camera_device_index: 0,
            usb_serial_port: String::new(),
            usb_serial_baud: 115200,
            wifi_interface: String::new(),
        }
    }
}

/// Central shared state - protected by parking_lot::Mutex
pub struct SharedState {
    pub extraction_pool: EntropyExtractionPool,
    pub pool: [u8; 32],
    pub display_pool: VecDeque<u8>,
    pub history_raw_entropy: VecDeque<f64>,
    pub history_whitened_entropy: VecDeque<f64>,
    pub source_metrics: HashMap<String, SourceMetrics>,
    pub estimated_true_entropy_bits: f64,
    pub credited_entropy_bits: f64,
    pub logs: VecDeque<String>,
    pub total_bytes: usize,
    pub sequence_id: u64,

    // Network
    pub net_mode: bool,
    pub uplink_url: String,

    // PQC Identity
    pub falcon_pk: Vec<u8>,
    pub falcon_sk: Vec<u8>,
    pub pqc_active: bool,

    // Harvesters
    pub harvester_states: HarvesterStates,
    pub health_testers: HashMap<String, NistHealthTester>,

    // P2P
    pub p2p_config: P2PConfig,

    // Auto-mint
    pub auto_mint_enabled: bool,
    pub last_auto_mint_ts: u64,

    // GPU (per-GPU independent tracking)
    pub gpu_cuda_available: bool,
    pub gpu_cuda_backend: String,
    pub gpu_ocl_available: bool,
    pub gpu_ocl_backend: String,

    // Guitar ESP32 sources
    pub guitar_states: HashMap<String, GuitarState>,

    // Headscale forwarding
    pub headscale: HeadscaleConfig,

    // Device manager
    pub device_config: DeviceConfig,

    // WiFi noise tracking
    pub wifi_active: bool,
    pub wifi_samples: u64,

    // USB serial tracking
    pub usb_serial_active: bool,
    pub usb_serial_bytes: u64,

    // Live mode
    pub live_mode: bool,

    // Mouse harvester lazy-start flag (rdev::listen causes OS-level lag if started at boot)
    pub mouse_harvester_started: bool,

    // Global Shannon entropy tracking (for GUI display)
    pub last_shannon: f64,
}

// ============================================================================
// CHAOSENGINE - Core Engine (Native Rust)
// ============================================================================

pub struct ChaosEngine {
    pub state: Arc<Mutex<SharedState>>,
    pub running: Arc<AtomicBool>,
    pub tx_entropy: Sender<(String, Vec<u8>)>,
}

// ============================================================================
// MIXER THREAD WITH NIST ENTROPY CREDITING
// Uses conditioned aggregate entropy for auto-mint decisions
// ============================================================================

fn start_mixer_thread(
    rx: Receiver<(String, Vec<u8>)>,
    state: Arc<Mutex<SharedState>>,
    running: Arc<AtomicBool>,
) {
    thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(500))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());

        let mut last_net_time = 0u64;
        let mut last_headscale_time = 0u64;

        while running.load(Ordering::Relaxed) {
            let (source, data) = match rx.recv_timeout(Duration::from_secs(1)) {
                Ok(d) => d,
                Err(_) => continue,
            };

            // Calculate entropy estimates (no lock needed)
            let raw_shannon = shannon_entropy(&data);
            let raw_min = conservative_min_entropy(&data);

            // NIST-credited entropy contribution
            let credited_bits = credit_entropy(data.len(), raw_min);

            // =============================================================
            // PHASE 1: Brief lock — state updates, collect config for Phase 2
            // =============================================================
            // Struct to hold everything we need after dropping the lock
            struct MixerExtracted {
                extracted: Vec<u8>,
                seq_id: u64,
                // Auto-mint prerequisites (None = don't mint)
                mint_pool: Option<[u8; 32]>,
                mint_falcon_sk: Option<Vec<u8>>,
                mint_falcon_pk: Option<Vec<u8>>,
                mint_agg_bits: f64,
                mint_now_ts: u64,
                // Network config snapshots
                net_target: Option<String>,     // uplink URL
                hs_url: Option<String>,         // headscale URL
                hs_fwd_count: u64,
                p2p_peers: Option<Vec<String>>, // P2P peer list
                p2p_hmac_key: Option<Vec<u8>>,  // P2P HMAC key
            }

            let phase2: Option<MixerExtracted> = {
                let mut lock = state.lock();

                // Feed to extraction pool with entropy tracking
                let extracted_opt = lock.extraction_pool.add_raw_bytes(&data, raw_min);

                // Update source metrics
                let metrics = lock.source_metrics.entry(source.clone()).or_default();
                metrics.samples += 1;
                metrics.raw_shannon = raw_shannon;
                metrics.min_entropy = raw_min;
                metrics.total_bits_contributed += credited_bits;
                metrics.avg_raw_entropy = if metrics.samples == 1 {
                    raw_shannon
                } else {
                    metrics.avg_raw_entropy * 0.95 + raw_shannon * 0.05
                };

                lock.estimated_true_entropy_bits += credited_bits;
                lock.credited_entropy_bits += credited_bits;
                lock.last_shannon = raw_shannon;

                // Update history
                if lock.history_raw_entropy.len() >= HISTORY_LEN {
                    lock.history_raw_entropy.pop_front();
                }
                lock.history_raw_entropy.push_back(raw_min);

                // Process extracted entropy
                if let Some(extracted) = extracted_opt {
                    let extracted_shannon = shannon_entropy(&extracted);

                    if lock.history_whitened_entropy.len() >= HISTORY_LEN {
                        lock.history_whitened_entropy.pop_front();
                    }
                    lock.history_whitened_entropy.push_back(extracted_shannon);

                    // Mix into pool using SHA-3
                    let mut pool_hasher = Sha3_256::new();
                    pool_hasher.update(&lock.pool);
                    pool_hasher.update(source.as_bytes());
                    pool_hasher.update(&extracted);
                    pool_hasher.update(&get_timestamp_nanos().to_le_bytes());
                    lock.pool = pool_hasher.finalize().into();

                    // Update display pool
                    for &b in extracted.iter() {
                        if lock.display_pool.len() >= POOL_SIZE {
                            lock.display_pool.pop_front();
                        }
                        lock.display_pool.push_back(b);
                    }

                    lock.total_bytes += extracted.len();
                    lock.sequence_id += 1;

                    // Log extraction (ASCII-safe)
                    let ts = chrono::Local::now().format("%H:%M:%S").to_string();
                    let msg = format!(
                        "[{}] EXTRACT #{} | {}->32 bytes | H_min:{:.2} | Credited:{:.0} bits | Src:{}",
                        ts, lock.extraction_pool.extractions_count,
                        EXTRACTION_POOL_SIZE, raw_min, credited_bits, source
                    );
                    if lock.logs.len() >= 500 { lock.logs.pop_front(); }
                    lock.logs.push_back(msg);

                    // --- Collect auto-mint prerequisites ---
                    let agg_bits = lock.extraction_pool.aggregate_credited_bits;
                    let should_mint = agg_bits >= MIN_ENTROPY_FOR_MINT
                        && raw_min > AUTO_MINT_THRESHOLD
                        && lock.pqc_active
                        && lock.auto_mint_enabled;

                    let (mint_pool, mint_fsk, mint_fpk, mint_agg, mint_ts) = if should_mint {
                        let now_ts = get_timestamp();
                        if lock.last_auto_mint_ts == 0 || (now_ts - lock.last_auto_mint_ts) >= 10 {
                            let ts = chrono::Local::now().format("%H:%M:%S").to_string();
                            let msg = format!(
                                "[{}] AUTO-MINT: Aggregate={:.0} bits, H_min={:.2}",
                                ts, agg_bits, raw_min
                            );
                            if lock.logs.len() >= 500 { lock.logs.pop_front(); }
                            lock.logs.push_back(msg);
                            (Some(lock.pool), Some(lock.falcon_sk.clone()),
                             Some(lock.falcon_pk.clone()), agg_bits, now_ts)
                        } else {
                            (None, None, None, 0.0, 0)
                        }
                    } else {
                        (None, None, None, 0.0, 0)
                    };

                    // --- Collect network config snapshots ---
                    let now = get_timestamp();

                    let net_target = if lock.net_mode && now > last_net_time {
                        last_net_time = now;
                        Some(lock.uplink_url.clone())
                    } else {
                        None
                    };

                    let hs_url = if lock.headscale.enabled && now > last_headscale_time + 5 {
                        last_headscale_time = now;
                        lock.headscale.forwarded_count += 1;
                        lock.headscale.last_forward_ts = now;
                        Some(format!("http://{}:{}/entropy",
                            lock.headscale.target_ip, lock.headscale.target_port))
                    } else {
                        None
                    };
                    let hs_fwd_count = lock.headscale.forwarded_count;

                    let (p2p_peers, p2p_hmac) = if lock.p2p_config.active && !lock.p2p_config.peers.is_empty() {
                        (Some(lock.p2p_config.peers.clone()),
                         lock.p2p_config.hmac_key.clone())
                    } else {
                        (None, None)
                    };

                    let seq_id = lock.sequence_id;

                    Some(MixerExtracted {
                        extracted,
                        seq_id,
                        mint_pool,
                        mint_falcon_sk: mint_fsk,
                        mint_falcon_pk: mint_fpk,
                        mint_agg_bits: mint_agg,
                        mint_now_ts: mint_ts,
                        net_target,
                        hs_url,
                        hs_fwd_count,
                        p2p_peers,
                        p2p_hmac_key: p2p_hmac,
                    })
                } else {
                    None
                }
            }; // === LOCK DROPS HERE ===

            // =============================================================
            // PHASE 2: No lock — expensive PQC keygen + network sends
            // =============================================================
            if let Some(mx) = phase2 {

                // --- AUTO-MINT (PQC keygen is CPU-heavy, runs WITHOUT lock) ---
                if let (Some(pool), Some(falcon_sk_bytes), Some(falcon_pk_bytes)) =
                    (mx.mint_pool, mx.mint_falcon_sk, mx.mint_falcon_pk)
                {
                    let (kyber_pk, kyber_sk) = kyber512::keypair();

                    let mut context_hasher = Sha3_256::new();
                    context_hasher.update(&pool);
                    context_hasher.update(kyber_pk.as_bytes());
                    let context = context_hasher.finalize();

                    if let Ok(falcon_secret) = falcon512::SecretKey::from_bytes(&falcon_sk_bytes) {
                        let signature = falcon512::detached_sign(&context, &falcon_secret);
                        let timestamp = get_timestamp();

                        let bundle = serde_json::json!({
                            "type": "NULLMAGNET_PQC_BUNDLE",
                            "version": "1.0",
                            "nist_compliant": true,
                            "requester": "RUST_AUTO",
                            "timestamp": timestamp,
                            "raw_min_entropy": raw_min,
                            "aggregate_credited_bits": mx.mint_agg_bits,
                            "kyber_pk": hex::encode(kyber_pk.as_bytes()),
                            "kyber_sk": hex::encode(kyber_sk.as_bytes()),
                            "falcon_sig": hex::encode(signature.as_bytes()),
                            "falcon_signer_pk": hex::encode(&falcon_pk_bytes),
                        });

                        let filename = format!("keys/key_{}_{}.json",
                            timestamp, hex::encode(&kyber_pk.as_bytes()[0..4]));
                        if let Ok(file) = fs::File::create(&filename) {
                            let _ = serde_json::to_writer_pretty(file, &bundle);
                        }

                        // Brief re-lock: update mint state + log
                        {
                            let mut lock = state.lock();
                            lock.extraction_pool.reset_aggregate_credits();
                            lock.credited_entropy_bits = 0.0;
                            lock.last_auto_mint_ts = mx.mint_now_ts;

                            let ts = chrono::Local::now().format("%H:%M:%S").to_string();
                            let msg = format!("[{}] VAULT: Saved {}", ts, filename);
                            if lock.logs.len() >= 500 { lock.logs.pop_front(); }
                            lock.logs.push_back(msg);
                        }
                    }
                }

                // --- NETWORK UPLINK (spawns async, no lock needed) ---
                if let Some(target) = mx.net_target {
                    let seq = mx.seq_id;
                    let source_clone = source.clone();
                    let c = client.clone();
                    let payload_hex = hex::encode(&mx.extracted[..]);
                    let payload_size = mx.extracted.len();

                    let digest = {
                        let mut hasher = Sha3_256::new();
                        hasher.update(&data);
                        hex::encode(hasher.finalize())
                    };

                    let ts_epoch = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_secs_f64();

                    let raw_min_copy = raw_min;
                    let raw_shannon_copy = raw_shannon;
                    let credited_copy = credited_bits;

                    thread::spawn(move || {
                        let _ = c.post(&target)
                            .json(&serde_json::json!({
                                "node": "nullmagnet_live",
                                "version": "1.0",
                                "nist_compliant": true,
                                "seq": seq,
                                "timestamp": get_timestamp(),
                                "ts_epoch": ts_epoch,
                                "entropy_estimate_raw_shannon": raw_shannon_copy,
                                "entropy_estimate_raw_min": raw_min_copy,
                                "credited_bits": credited_copy,
                                "health": "OK",
                                "source": source_clone,
                                "metrics": {"size": payload_size},
                                "payload_hex": payload_hex,
                                "digest": digest
                            }))
                            .send();
                    });
                }

                // --- HEADSCALE FORWARDING (spawns async, no lock needed) ---
                if let Some(hs_url) = mx.hs_url {
                    let payload_hex = hex::encode(&mx.extracted[..]);
                    let seq = mx.seq_id;
                    let c = client.clone();
                    let fwd_count = mx.hs_fwd_count;

                    thread::spawn(move || {
                        let _ = c.post(&hs_url)
                            .json(&serde_json::json!({
                                "node": "nullmagnet_live",
                                "version": "1.0",
                                "seq": seq,
                                "timestamp": get_timestamp(),
                                "payload_hex": payload_hex,
                                "forward_count": fwd_count,
                            }))
                            .send();
                    });
                }

                // --- P2P DISTRIBUTION (spawns async, no lock needed) ---
                if let Some(peers) = mx.p2p_peers {
                    let payload_hex = hex::encode(&mx.extracted[..]);
                    let seq = mx.seq_id;
                    let c = client.clone();
                    let hmac_key = mx.p2p_hmac_key;

                    thread::spawn(move || {
                        let timestamp = get_timestamp() as i64;

                        for peer in peers {
                            let url = format!("http://{}", peer);

                            let mut payload = serde_json::json!({
                                "node_id": "nullmagnet_live",
                                "seq": seq,
                                "timestamp": timestamp,
                                "payload_hex": payload_hex,
                                "sources": "mixed",
                            });

                            // Add HMAC if key is configured
                            if let Some(ref key) = hmac_key {
                                let payload_bytes = hex::decode(&payload_hex).unwrap_or_default();
                                let mut mac = HmacSha256::new_from_slice(key).unwrap();
                                mac.update(b"nullmagnet_live|");
                                mac.update(&seq.to_le_bytes());
                                mac.update(b"|");
                                mac.update(&timestamp.to_le_bytes());
                                mac.update(b"|mixed|");
                                mac.update(&payload_bytes);
                                let mac_hex = hex::encode(mac.finalize().into_bytes());
                                payload["mac_hex"] = serde_json::Value::String(mac_hex);
                            }

                            let _ = c.post(&url).json(&payload).send();
                        }
                    });
                }
            }
        }
    });
}

// ============================================================================
// NATIVE RUST IMPLEMENTATION
// ============================================================================

impl ChaosEngine {
    pub fn new_native() -> Self {
        let (tx, rx) = bounded(1000);
        let _ = fs::create_dir_all("keys");

        let (pk, sk) = falcon512::keypair();
        let pqc_active = true;

        let mut display_pool = VecDeque::with_capacity(POOL_SIZE);
        display_pool.extend(vec![0u8; POOL_SIZE]);

        // Detect GPUs independently
        let (cuda_avail, cuda_backend) = harvesters::detect_gpu_cuda();
        let (ocl_avail, ocl_backend) = harvesters::detect_gpu_opencl();

        // Initialize guitar states
        let mut guitar_states = HashMap::new();
        for (name, data_port, ctrl_port) in &[
            ("Spectra",   5005u16, 5056u16),
            ("Neptonius", 5006u16, 5057u16),
            ("Thalyn",    5007u16, 5058u16),
        ] {
            guitar_states.insert(name.to_string(), GuitarState {
                name: name.to_string(),
                data_port: *data_port,
                ctrl_port: *ctrl_port,
                enabled: true,
                packets_received: 0,
                bytes_received: 0,
            });
        }

        let state = Arc::new(Mutex::new(SharedState {
            extraction_pool: EntropyExtractionPool::new(),
            pool: [0u8; 32],
            display_pool,
            history_raw_entropy: VecDeque::from(vec![0.0; HISTORY_LEN]),
            history_whitened_entropy: VecDeque::from(vec![0.0; HISTORY_LEN]),
            source_metrics: HashMap::new(),
            estimated_true_entropy_bits: 0.0,
            credited_entropy_bits: 0.0,
            logs: VecDeque::from(vec![
                format!("ENGINE: NullMagnet Live Core v1.0 (NIST SP 800-90B)"),
                format!("CONFIG: RCT_CUTOFF={}, APT_WINDOW={}, APT_CUTOFF={}",
                    NIST_RCT_CUTOFF, NIST_APT_WINDOW, NIST_APT_CUTOFF),
                format!("GPU-CUDA: {} ({})", if cuda_avail { "Available" } else { "N/A" }, cuda_backend),
                format!("GPU-OCL:  {} ({})", if ocl_avail { "Available" } else { "N/A" }, ocl_backend),
            ]),
            total_bytes: 0,
            net_mode: true,
            uplink_url: "http://192.168.1.19:8000/ingest".to_string(),
            sequence_id: 0,
            falcon_pk: pk.as_bytes().to_vec(),
            falcon_sk: sk.as_bytes().to_vec(),
            pqc_active,
            harvester_states: HarvesterStates::default(),
            health_testers: HashMap::new(),
            p2p_config: P2PConfig::default(),
            auto_mint_enabled: false,
            last_auto_mint_ts: 0,
            gpu_cuda_available: cuda_avail,
            gpu_cuda_backend: cuda_backend,
            gpu_ocl_available: ocl_avail,
            gpu_ocl_backend: ocl_backend,
            guitar_states,
            headscale: HeadscaleConfig::default(),
            device_config: DeviceConfig::default(),
            wifi_active: false,
            wifi_samples: 0,
            usb_serial_active: false,
            usb_serial_bytes: 0,
            live_mode: false,
            mouse_harvester_started: false,
            last_shannon: 0.0,
        }));

        {
            let mut lock = state.lock();
            let ts = chrono::Local::now().format("%H:%M:%S").to_string();
            lock.logs.push_back(format!(
                "[{}] IDENTITY: Falcon-512 Session Key Generated", ts));
            lock.logs.push_back(format!(
                "[{}] EXTRACTION: {}->32 byte conditioning (SHA-256)", ts, EXTRACTION_POOL_SIZE));
            lock.logs.push_back(format!(
                "[{}] STARTUP: {} samples discarded per source", ts, STARTUP_DISCARD_SAMPLES));
            lock.logs.push_back(format!(
                "[{}] GUITARS: Spectra:5005 Neptonius:5006 Thalyn:5007 (UDP)", ts));
            lock.logs.push_back(format!(
                "[{}] HEADSCALE: Aoi Midori @ 100.64.0.15:8100", ts));
        }

        let running = Arc::new(AtomicBool::new(true));

        // Start core threads
        start_mixer_thread(rx, state.clone(), running.clone());
        harvesters::start_p2p_server(tx.clone(), state.clone(), running.clone());

        // Start standard harvesters
        // NOTE: Mouse harvester is NOT started here. rdev::listen() installs
        // a global OS input hook (SetWindowsHookEx) that intercepts all mouse
        // events system-wide, causing lag. Started lazily via toggle_harvester().
        harvesters::start_trng_harvester(tx.clone(), running.clone(), state.clone());
        harvesters::start_audio_harvester(tx.clone(), running.clone(), state.clone());
        harvesters::start_system_harvester(tx.clone(), running.clone(), state.clone());
        harvesters::start_video_harvester(tx.clone(), running.clone(), state.clone());

        // Start per-GPU harvesters (independent threads)
        harvesters::start_gpu_cuda_harvester(tx.clone(), running.clone(), state.clone());
        harvesters::start_gpu_ocl_harvester(tx.clone(), running.clone(), state.clone());

        // Start new NullMagnet Live harvesters
        harvesters::start_guitar_udp_listener(tx.clone(), running.clone(), state.clone());
        harvesters::start_wifi_harvester(tx.clone(), running.clone(), state.clone());
        harvesters::start_usb_serial_harvester(tx.clone(), running.clone(), state.clone());
        harvesters::start_headscale_forwarder(state.clone(), running.clone());

        ChaosEngine {
            state,
            running,
            tx_entropy: tx,
        }
    }

    // ========================================================================
    // HARVESTER TOGGLES
    // ========================================================================

    pub fn toggle_harvester_internal(&self, name: &str, active: bool) {
        let mut lock = self.state.lock();
        match name.to_uppercase().as_str() {
            "TRNG" | "HARDWARE/TRNG"          => lock.harvester_states.trng = active,
            "AUDIO" | "AUDIO (MIC)"           => lock.harvester_states.audio = active,
            "SYS" | "SYSTEM" | "SYSTEM/CPU"   => lock.harvester_states.system = active,
            "MOUSE" | "HID (MOUSE)"           => {
                lock.harvester_states.mouse = active;
                // Lazy-start: only spawn rdev::listen() the first time mouse is enabled
                // This avoids the OS-level input hook lag at startup
                if active && !lock.mouse_harvester_started {
                    lock.mouse_harvester_started = true;
                    let tx = self.tx_entropy.clone();
                    let running = self.running.clone();
                    let state_clone = self.state.clone();
                    drop(lock);  // Release mutex before spawning thread
                    harvesters::start_mouse_harvester(tx, running, state_clone);
                    let mut lock = self.state.lock();
                    let ts = chrono::Local::now().format("%H:%M:%S").to_string();
                    let msg = format!("[{}] Toggle: {} -> Active (hook started)", ts, name);
                    if lock.logs.len() >= 500 { lock.logs.pop_front(); }
                    lock.logs.push_back(msg);
                    return;
                }
            }
            "VIDEO" | "VIDEO (CAM)"           => lock.harvester_states.video = active,
            "GPU_CUDA" | "GPU (CUDA)"         => lock.harvester_states.gpu_cuda = active,
            "GPU_OCL" | "GPU (OPENCL)"        => lock.harvester_states.gpu_ocl = active,
            "WIFI" | "WIFI NOISE"             => lock.harvester_states.wifi = active,
            "USB_SERIAL" | "USB SERIAL"       => lock.harvester_states.usb_serial = active,
            other => {
                // Check guitar names: GUITAR_SPECTRA, GUITAR_NEPTONIUS, etc.
                if other.starts_with("GUITAR_") {
                    let gname = other.trim_start_matches("GUITAR_");
                    // Find guitar by case-insensitive name
                    let key = lock.guitar_states.keys()
                        .find(|k| k.to_uppercase() == gname)
                        .cloned();
                    if let Some(key) = key {
                        if let Some(gs) = lock.guitar_states.get_mut(&key) {
                            gs.enabled = active;
                        }
                    }
                }
            }
        }

        let status = if active { "Active" } else { "Inactive" };
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] Toggle: {} -> {}", ts, name, status);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }

    // ========================================================================
    // NETWORK CONTROLS
    // ========================================================================

    pub fn toggle_uplink(&self, active: bool) {
        let mut lock = self.state.lock();
        lock.net_mode = active;

        let status = if active { "ENABLED" } else { "PAUSED" };
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] Network Uplink -> {}", ts, status);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }

    pub fn set_network_target(&self, ip: String) {
        let mut lock = self.state.lock();
        lock.uplink_url = format!("http://{}:8000/ingest", ip);

        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] NET: Target set to {}", ts, ip);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }

    pub fn set_uplink_target(&self, ip: String, port: u16) {
        let port = if port == 0 { 8000 } else { port };
        let mut lock = self.state.lock();
        lock.uplink_url = format!("http://{}:{}/entropy", ip, port);

        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] NET: Uplink target set to {}:{}", ts, ip, port);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }

    pub fn toggle_headscale(&self, active: bool) {
        let mut lock = self.state.lock();
        lock.headscale.enabled = active;

        let status = if active { "ENABLED" } else { "DISABLED" };
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] Headscale (Aoi Midori) -> {} ({}:{})",
            ts, status, lock.headscale.target_ip, lock.headscale.target_port);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }

    pub fn set_headscale_target(&self, ip: String, port: u16) {
        let port = if port == 0 { 8100 } else { port };
        let mut lock = self.state.lock();
        lock.headscale.target_ip = ip.clone();
        lock.headscale.target_port = port;

        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] HEADSCALE: Target set to Aoi Midori @ {}:{}", ts, ip, port);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }

    // ========================================================================
    // P2P CONTROLS
    // ========================================================================

    pub fn toggle_p2p(&self, active: bool) {
        let mut lock = self.state.lock();
        lock.p2p_config.active = active;

        let status = if active { "ENABLED" } else { "PAUSED" };
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] P2P Mode -> {}", ts, status);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }

    pub fn set_p2p_port(&self, port: u16) {
        let mut lock = self.state.lock();
        lock.p2p_config.listen_port = port;

        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] P2P: Listen port set to {}", ts, port);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }

    pub fn set_p2p_hmac_key(&self, key_hex: String) -> Result<(), String> {
        let key = hex::decode(key_hex.trim())
            .map_err(|_| "Invalid hex key".to_string())?;

        if key.len() != 32 {
            return Err("HMAC key must be 32 bytes".to_string());
        }

        let mut lock = self.state.lock();
        lock.p2p_config.hmac_key = Some(key);

        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] P2P: HMAC authentication ENABLED", ts);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);

        Ok(())
    }

    pub fn add_peer(&self, peer_addr: String) {
        let mut lock = self.state.lock();
        if !lock.p2p_config.peers.contains(&peer_addr) {
            lock.p2p_config.peers.push(peer_addr.clone());

            let ts = chrono::Local::now().format("%H:%M:%S").to_string();
            let msg = format!("[{}] P2P: Added peer {}", ts, peer_addr);
            if lock.logs.len() >= 500 { lock.logs.pop_front(); }
            lock.logs.push_back(msg);
        }
    }

    // ========================================================================
    // PQC VAULT CONTROLS
    // ========================================================================

    pub fn set_auto_mint(&self, enabled: bool) {
        let mut lock = self.state.lock();
        lock.auto_mint_enabled = enabled;

        let status = if enabled { "ENABLED" } else { "DISABLED" };
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] AUTO-MINT -> {} (threshold: {} credited bits)",
            ts, status, MIN_ENTROPY_FOR_MINT as u32);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }

    pub fn mint_pqc_bundle_internal(&self, requester: &str) -> Result<String, String> {
        let requester = if requester.is_empty() { "LOCAL" } else { requester };
        let mut lock = self.state.lock();

        if !lock.pqc_active {
            return Ok("Error: PQC Engine Offline".to_string());
        }

        // Check conditioned aggregate entropy threshold
        let agg_bits = lock.extraction_pool.aggregate_credited_bits;
        if agg_bits < MIN_ENTROPY_FOR_MINT {
            return Ok(format!(
                "Error: Insufficient entropy ({:.0}/{} aggregate credited bits)",
                agg_bits, MIN_ENTROPY_FOR_MINT as u32
            ));
        }

        let (kyber_pk, kyber_sk) = kyber512::keypair();

        let mut context_hasher = Sha3_256::new();
        context_hasher.update(&lock.pool);
        context_hasher.update(kyber_pk.as_bytes());
        let context = context_hasher.finalize();

        let falcon_secret = falcon512::SecretKey::from_bytes(&lock.falcon_sk)
            .map_err(|e| format!("Falcon key error: {}", e))?;
        let signature = falcon512::detached_sign(&context, &falcon_secret);
        let timestamp = get_timestamp();

        let bundle = serde_json::json!({
            "type": "NULLMAGNET_PQC_BUNDLE",
            "version": "1.0",
            "nist_compliant": true,
            "requester": requester,
            "timestamp": timestamp,
            "aggregate_credited_bits": agg_bits,
            "kyber_pk": hex::encode(kyber_pk.as_bytes()),
            "kyber_sk": hex::encode(kyber_sk.as_bytes()),
            "falcon_sig": hex::encode(signature.as_bytes()),
            "falcon_signer_pk": hex::encode(&lock.falcon_pk),
        });

        let filename = format!("keys/key_{}_{}.json",
            timestamp, hex::encode(&kyber_pk.as_bytes()[0..4]));
        if let Ok(file) = fs::File::create(&filename) {
            let _ = serde_json::to_writer_pretty(file, &bundle);
        }

        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] VAULT: Saved {} (aggregate: {:.0} bits)", ts, filename, agg_bits);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);

        // Reset aggregate credited bits after mint
        lock.extraction_pool.reset_aggregate_credits();
        lock.credited_entropy_bits = 0.0;

        Ok(format!("Generated {}", filename))
    }

    // ========================================================================
    // HEALTH TEST CONTROLS
    // ========================================================================

    pub fn trigger_health_test(&self, source: String) {
        let mut lock = self.state.lock();
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] ON-DEMAND: Triggering health test for {}", ts, source);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
        // Individual harvesters manage their own health testers;
        // this logs the intent. Actual reset happens in harvester threads.
    }

    // ========================================================================
    // DEVICE MANAGER CONTROLS
    // ========================================================================

    pub fn set_audio_device(&self, index: usize) {
        let mut lock = self.state.lock();
        lock.device_config.audio_device_index = Some(index);
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] DEVICE: Audio device set to index {}", ts, index);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }

    pub fn set_audio_gain(&self, gain: f64) {
        let mut lock = self.state.lock();
        lock.device_config.audio_gain = gain.max(0.1).min(10.0);
    }

    pub fn set_camera_device(&self, index: usize) {
        let mut lock = self.state.lock();
        lock.device_config.camera_device_index = index;
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] DEVICE: Camera set to index {}", ts, index);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }

    pub fn set_usb_serial_port(&self, port: String, baud: u32) {
        let mut lock = self.state.lock();
        lock.device_config.usb_serial_port = port.clone();
        lock.device_config.usb_serial_baud = baud;
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] DEVICE: USB serial set to {} @ {}", ts, port, baud);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }

    pub fn set_wifi_interface(&self, iface: String) {
        let mut lock = self.state.lock();
        lock.device_config.wifi_interface = iface.clone();
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] DEVICE: WiFi interface set to {}", ts, iface);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }

    // ========================================================================
    // LIVE MODE
    // ========================================================================

    pub fn set_live_mode(&self, active: bool) {
        let mut lock = self.state.lock();
        lock.live_mode = active;

        // Live Mode enables/disables all safe harvesters
        // (Mouse excluded - rdev global hook causes system-wide input lag)
        lock.harvester_states.trng = active;
        lock.harvester_states.audio = active;
        lock.harvester_states.system = active;
        lock.harvester_states.video = active;
        lock.harvester_states.wifi = active;
        lock.harvester_states.usb_serial = active;

        // Enable GPUs if available
        if lock.gpu_cuda_available { lock.harvester_states.gpu_cuda = active; }
        if lock.gpu_ocl_available  { lock.harvester_states.gpu_ocl = active; }

        // Enable all guitars
        for gs in lock.guitar_states.values_mut() {
            gs.enabled = active;
        }

        let status = if active { "ON — all harvesters enabled" } else { "OFF — all harvesters disabled" };
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let msg = format!("[{}] LIVE MODE -> {}", ts, status);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }

    // ========================================================================
    // METRICS (consumed by Tauri GUI every frame)
    // ========================================================================

    pub fn get_metrics_internal(&self) -> Result<String, String> {
        // Regular lock - called only 4x/sec (250ms interval) so blocking is fine.
        // The actual perf fixes are in the harvesters (atomics, try_lock there).
        let lock = self.state.lock();

        let current_raw = lock.history_raw_entropy.back().copied().unwrap_or(0.0);
        let current_whitened = lock.history_whitened_entropy.back().copied().unwrap_or(0.0);
        let current_shannon = lock.last_shannon;
        // Conditioned H_min = raw H_min * 0.85 (NIST conditioning factor)
        let conditioned_hmin = current_raw * NIST_CONDITIONING_FACTOR;

        let source_quality: HashMap<String, serde_json::Value> = lock.source_metrics.iter()
            .map(|(name, m)| {
                (name.clone(), serde_json::json!({
                    "raw_shannon": m.raw_shannon,
                    "min_entropy": m.min_entropy,
                    "avg_entropy": m.avg_raw_entropy,
                    "samples": m.samples,
                    "total_bits": m.total_bits_contributed,
                    "health_state": m.health_state,
                }))
            })
            .collect();

        // Build guitar packet counts
        let mut guitar_metrics = serde_json::Map::new();
        for (gname, gs) in &lock.guitar_states {
            guitar_metrics.insert(
                format!("guitar_{}_packets", gname.to_lowercase()),
                serde_json::json!(gs.packets_received),
            );
            guitar_metrics.insert(
                format!("guitar_{}_bytes", gname.to_lowercase()),
                serde_json::json!(gs.bytes_received),
            );
        }

        let agg_bits = lock.extraction_pool.aggregate_credited_bits;

        let mut metrics = serde_json::json!({
            "version": "1.0",
            "nist_compliant": true,
            "pool_hex": hex::encode(lock.pool).to_uppercase(),
            "total_bytes": lock.total_bytes,
            "current_entropy": current_raw,
            "current_raw_entropy": current_raw,
            "current_shannon": current_shannon,
            "conditioned_hmin": conditioned_hmin,
            "current_whitened_entropy": current_whitened,
            "estimated_true_bits": lock.estimated_true_entropy_bits,
            "credited_entropy_bits": lock.credited_entropy_bits,
            "aggregate_credited_bits": agg_bits,
            "min_entropy_for_mint": MIN_ENTROPY_FOR_MINT,

            // Extraction pool metrics
            "extraction_pool_fill": lock.extraction_pool.fill_percentage(),
            "extraction_pool_accumulated": lock.extraction_pool.accumulated_bytes(),
            "extractions_count": lock.extraction_pool.extractions_count,
            "total_raw_consumed": lock.extraction_pool.total_raw_consumed,
            "total_extracted_bytes": lock.extraction_pool.total_extracted_bytes,

            // NIST configuration
            "nist_rct_cutoff": NIST_RCT_CUTOFF,
            "nist_apt_window": NIST_APT_WINDOW,
            "nist_apt_cutoff": NIST_APT_CUTOFF,
            "nist_conditioning_factor": NIST_CONDITIONING_FACTOR,
            "startup_discard_samples": STARTUP_DISCARD_SAMPLES,

            "source_quality": source_quality,
            "history": lock.history_raw_entropy.iter().collect::<Vec<_>>(),
            "history_raw": lock.history_raw_entropy.iter().collect::<Vec<_>>(),
            "history_whitened": lock.history_whitened_entropy.iter().collect::<Vec<_>>(),
            "logs": lock.logs.iter().collect::<Vec<_>>(),
            "net_mode": lock.net_mode,
            "pqc_ready": lock.pqc_active,

            // Per-GPU status (independent)
            "gpu_cuda_available": lock.gpu_cuda_available,
            "gpu_cuda_backend": lock.gpu_cuda_backend.clone(),
            "gpu_cuda_enabled": lock.harvester_states.gpu_cuda,
            "gpu_ocl_available": lock.gpu_ocl_available,
            "gpu_ocl_backend": lock.gpu_ocl_backend.clone(),
            "gpu_ocl_enabled": lock.harvester_states.gpu_ocl,

            // Backward compat: single gpu_available field
            "gpu_available": lock.gpu_cuda_available || lock.gpu_ocl_available,
            "gpu_backend": if lock.gpu_cuda_available { &lock.gpu_cuda_backend }
                           else { &lock.gpu_ocl_backend },
            "gpu_enabled": lock.harvester_states.gpu_cuda || lock.harvester_states.gpu_ocl,

            // P2P metrics
            "p2p_active": lock.p2p_config.active,
            "p2p_port": lock.p2p_config.listen_port,
            "p2p_peer_count": lock.p2p_config.peers.len(),
            "p2p_received_count": lock.p2p_config.received_count,
            "p2p_hmac_enabled": lock.p2p_config.hmac_key.is_some(),
            "auto_mint_enabled": lock.auto_mint_enabled,

            // Headscale
            "headscale_active": lock.headscale.enabled,
            "headscale_forwarded": lock.headscale.forwarded_count,

            // WiFi
            "wifi_active": lock.wifi_active,
            "wifi_samples": lock.wifi_samples,

            // USB Serial
            "usb_serial_active": lock.usb_serial_active,
            "usb_serial_bytes": lock.usb_serial_bytes,

            // Live mode
            "live_mode": lock.live_mode,

            // Harvester enable states (for frontend checkbox sync)
            "harvester_trng": lock.harvester_states.trng,
            "harvester_audio": lock.harvester_states.audio,
            "harvester_system": lock.harvester_states.system,
            "harvester_mouse": lock.harvester_states.mouse,
            "harvester_video": lock.harvester_states.video,
            "harvester_wifi": lock.harvester_states.wifi,
            "harvester_usb_serial": lock.harvester_states.usb_serial,
        });

        // Merge guitar metrics into top level
        if let Some(obj) = metrics.as_object_mut() {
            for (k, v) in guitar_metrics {
                obj.insert(k, v);
            }
        }

        Ok(metrics.to_string())
    }

    // ========================================================================
    // SHUTDOWN
    // ========================================================================

    pub fn shutdown_internal(&self) {
        self.running.store(false, Ordering::Relaxed);
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        let mut lock = self.state.lock();

        // FIPS 140-3: Zeroize all secret key material on shutdown
        lock.falcon_sk.zeroize();
        lock.pool.zeroize();
        if let Some(ref mut hmac_key) = lock.p2p_config.hmac_key {
            hmac_key.zeroize();
        }

        let msg = format!("[{}] ENGINE: Shutdown + key zeroization complete", ts);
        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
        lock.logs.push_back(msg);
    }
}

/// FIPS 140-3: Ensure secret key material is zeroized when SharedState is dropped
impl Drop for SharedState {
    fn drop(&mut self) {
        self.falcon_sk.zeroize();
        self.pool.zeroize();
        if let Some(ref mut hmac_key) = self.p2p_config.hmac_key {
            hmac_key.zeroize();
        }
    }
}

// ============================================================================
// MODULE REGISTRATION (Tauri - no PyO3 needed)
// ChaosEngine is constructed directly in main.rs via new_native()
// ============================================================================
