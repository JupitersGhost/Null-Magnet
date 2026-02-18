//! NullMagnet Live - harvesters.rs
//! Jupiter Labs - All entropy harvester threads
//!
//! Each harvester runs in its own thread, performs NIST health testing
//! locally, and sends passed samples to the mixer via crossbeam channel.
//!
//! Harvesters:
//!   Standard:  TRNG, Audio, System, Mouse, Video
//!   GPU:       CUDA (independent thread), OpenCL (independent thread)
//!   Live:      Guitar ESP32 UDP (Spectra, Neptonius, Thalyn)
//!   Extended:  WiFi noise, USB serial
//!   Network:   P2P server (HMAC auth), Headscale forwarder (Aoi Midori)

use std::sync::{Arc, atomic::{AtomicBool, AtomicUsize, Ordering}};
use parking_lot::Mutex;
use crossbeam_channel::Sender;
use std::thread;
use std::time::{Duration, Instant};
use sha2::Digest as Sha2Digest;
use sha3::Sha3_256;
use hmac::Mac;

use crate::SharedState;
use crate::HmacSha256;
use crate::entropy::{
    NistHealthTester,
    get_timestamp, get_timestamp_nanos,
};

// ============================================================================
// GPU ENTROPY - CUDA (NVIDIA) - Independent Thread
// ============================================================================

#[cfg(feature = "gpu-cuda")]
mod cuda_entropy {
    pub fn harvest_cuda(size: usize) -> Option<Vec<u8>> {
        use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

        const KERNEL: &str = r#"
            extern "C" __global__ void race_entropy(int *out, int n, int iters) {
                __shared__ int s[256];
                int t = threadIdx.x;
                s[t % 256] = 0;
                __syncthreads();
                for (int i = 0; i < iters; i++) {
                    s[(t + i) % 256] += t;
                    s[(t + i) % 256] ^= (i * 37);
                    __threadfence_block();
                }
                __syncthreads();
                if (t < n) out[t] = s[t % 256] ^ (t * 0x9E3779B9);
            }
        "#;

        let device = CudaDevice::new(0).ok()?;
        let ptx = cudarc::nvrtc::compile_ptx(KERNEL).ok()?;
        device.load_ptx(ptx, "entropy", &["race_entropy"]).ok()?;

        let mut output = device.alloc_zeros::<i32>(size).ok()?;

        let cfg = LaunchConfig {
            block_dim: (256, 1, 1),
            grid_dim: (((size as u32) + 255) / 256, 1, 1),
            shared_mem_bytes: 1024,
        };

        unsafe {
            let f = device.get_func("entropy", "race_entropy")?;
            f.launch(cfg, (&mut output, size as i32, 100i32)).ok()?;
        }

        device.synchronize().ok()?;
        let host = device.dtoh_sync_copy(&output).ok()?;

        // Extract low bits (highest entropy from race conditions)
        let bytes: Vec<u8> = host.iter()
            .flat_map(|&x| [(x & 0xFF) as u8, ((x >> 8) & 0xFF) as u8])
            .take(size)
            .collect();

        Some(bytes)
    }

    pub fn is_available() -> bool {
        cudarc::driver::CudaDevice::new(0).is_ok()
    }
}

#[cfg(feature = "gpu-opencl")]
mod opencl_entropy {
    pub fn harvest_opencl(size: usize) -> Option<Vec<u8>> {
        use ocl::{Buffer, Context, Device, Kernel, Platform, Program, Queue};

        const KERNEL: &str = r#"
            __kernel void race_entropy(__global int *out, int n, int iters) {
                __local int s[256];
                int t = get_local_id(0);
                int g = get_global_id(0);
                s[t % 256] = 0;
                barrier(CLK_LOCAL_MEM_FENCE);
                for (int i = 0; i < iters; i++) {
                    s[(t + i) % 256] += t;
                    s[(t + i) % 256] ^= (i * 37);
                    mem_fence(CLK_LOCAL_MEM_FENCE);
                }
                barrier(CLK_LOCAL_MEM_FENCE);
                if (g < n) out[g] = s[t % 256] ^ (t * 0x9E3779B9);
            }
        "#;

        let platform = Platform::default();
        let device = Device::first(platform).ok()?;
        let context = Context::builder().platform(platform).devices(device).build().ok()?;
        let queue = Queue::new(&context, device, None).ok()?;
        let program = Program::builder().src(KERNEL).devices(device).build(&context).ok()?;

        let buffer = Buffer::<i32>::builder().queue(queue.clone()).len(size).build().ok()?;

        let kernel = Kernel::builder()
            .program(&program)
            .name("race_entropy")
            .queue(queue.clone())
            .global_work_size(((size + 255) / 256) * 256)
            .local_work_size(256)
            .arg(&buffer)
            .arg(size as i32)
            .arg(100i32)
            .build().ok()?;

        unsafe { kernel.enq().ok()?; }

        let mut host = vec![0i32; size];
        buffer.read(&mut host).enq().ok()?;
        queue.finish().ok()?;

        let bytes: Vec<u8> = host.iter()
            .flat_map(|&x| [(x & 0xFF) as u8, ((x >> 8) & 0xFF) as u8])
            .take(size)
            .collect();

        Some(bytes)
    }

    pub fn is_available() -> bool {
        use ocl::{Device, Platform};
        Platform::default().try_into().ok()
            .and_then(|p: Platform| Device::first(p).ok())
            .is_some()
    }
}

// ============================================================================
// GPU DETECTION (per-GPU, independent)
// ============================================================================

pub fn detect_gpu_cuda() -> (bool, String) {
    #[cfg(feature = "gpu-cuda")]
    {
        if cuda_entropy::is_available() {
            return (true, "CUDA (NVIDIA)".to_string());
        }
    }
    (false, "None".to_string())
}

pub fn detect_gpu_opencl() -> (bool, String) {
    #[cfg(feature = "gpu-opencl")]
    {
        if opencl_entropy::is_available() {
            return (true, "OpenCL (AMD/Intel)".to_string());
        }
    }
    (false, "None".to_string())
}

// ============================================================================
// STANDARD HARVESTERS
// Each runs in its own thread with a local NistHealthTester
// ============================================================================

// --- TRNG (OS Random Number Generator) ---

pub fn start_trng_harvester(
    tx: Sender<(String, Vec<u8>)>,
    running: Arc<AtomicBool>,
    state: Arc<Mutex<SharedState>>,
) {
    thread::spawn(move || {
        use rand::prelude::*;
        let mut rng = rand::rngs::OsRng;
        let mut health_tester = NistHealthTester::new();
        health_tester.start();

        while running.load(Ordering::Relaxed) {
            let enabled = state.try_lock()
                .map(|l| l.harvester_states.trng)
                .unwrap_or(false);
            if enabled {
                let mut buf = [0u8; 1024];
                rng.fill_bytes(&mut buf);

                let passed = health_tester.process_batch(&buf);
                if !passed.is_empty() {
                    let _ = tx.try_send(("TRNG".to_string(), passed));
                }

                // Update health state
                if let Some(mut lock) = state.try_lock() {
                    let metrics = lock.source_metrics.entry("TRNG".to_string()).or_default();
                    metrics.health_state = health_tester.state_name().to_string();
                }
            }
            thread::sleep(Duration::from_secs(1));
        }
    });
}

// --- Audio (Microphone ADC Noise) ---

pub fn start_audio_harvester(
    tx: Sender<(String, Vec<u8>)>,
    running: Arc<AtomicBool>,
    state: Arc<Mutex<SharedState>>,
) {
    thread::spawn(move || {
        use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

        let host = cpal::default_host();
        let device = match host.default_input_device() {
            Some(d) => d,
            None => return,
        };

        let config = match device.default_input_config() {
            Ok(c) => c,
            Err(_) => return,
        };

        let tx_clone = tx.clone();
        let running_stream = running.clone();
        let state_clone = state.clone();

        let last_send = Arc::new(Mutex::new(Instant::now()));
        let last_send_clone = last_send.clone();
        let health_tester = Arc::new(Mutex::new({
            let mut h = NistHealthTester::new();
            h.start();
            h
        }));
        let health_clone = health_tester.clone();

        // Lock-free enabled check for the hot audio callback path
        let audio_enabled = Arc::new(AtomicBool::new(false));
        let audio_enabled_cb = audio_enabled.clone();
        // Cache gain as atomic bits to avoid locking in callback
        use std::sync::atomic::AtomicU64;
        let gain_bits = Arc::new(AtomicU64::new(1.0f64.to_bits()));
        let gain_bits_cb = gain_bits.clone();

        // Poller thread: syncs enabled + gain from SharedState every 200ms
        let state_poll = state.clone();
        let running_poll = running.clone();
        let audio_enabled_poll = audio_enabled.clone();
        let gain_bits_poll = gain_bits.clone();
        thread::spawn(move || {
            while running_poll.load(Ordering::Relaxed) {
                if let Some(lock) = state_poll.try_lock() {
                    audio_enabled_poll.store(lock.harvester_states.audio, Ordering::Relaxed);
                    gain_bits_poll.store(lock.device_config.audio_gain.to_bits(), Ordering::Relaxed);
                }
                thread::sleep(Duration::from_millis(200));
            }
        });

        let stream = device.build_input_stream(
            &config.into(),
            move |data: &[f32], _: &_| {
                if !running_stream.load(Ordering::Relaxed) { return; }

                // Fast atomic check - no mutex lock in hot path
                if !audio_enabled_cb.load(Ordering::Relaxed) { return; }

                // Throttle: only process every 200ms
                let mut last = match last_send_clone.try_lock() {
                    Some(l) => l,
                    None => return,  // Skip if busy
                };
                if last.elapsed() < Duration::from_millis(200) {
                    return;
                }
                *last = Instant::now();
                drop(last);

                // Read cached gain atomically - no mutex
                let gain = f64::from_bits(gain_bits_cb.load(Ordering::Relaxed));

                // Extract LSBs (highest entropy in ADC noise)
                let sample_limit = data.len().min(256);
                let mut bytes = Vec::with_capacity(sample_limit);

                for &sample in data.iter().take(sample_limit).step_by(2) {
                    let amplified = sample * gain as f32;
                    let bits = amplified.to_bits();
                    bytes.push((bits & 0xFF) as u8);
                }

                let nanos = get_timestamp_nanos();
                bytes.extend_from_slice(&nanos.to_le_bytes());

                // Health test - try_lock to never block audio thread
                if let Some(mut ht) = health_clone.try_lock() {
                    let passed = ht.process_batch(&bytes);
                    if !passed.is_empty() {
                        let _ = tx_clone.try_send(("AUDIO".to_string(), passed));
                    }
                    // Update health state via try_lock
                    if let Some(mut lock) = state_clone.try_lock() {
                        let metrics = lock.source_metrics.entry("AUDIO".to_string()).or_default();
                        metrics.health_state = ht.state_name().to_string();
                    }
                }
            },
            |_| {},
            None,
        );

        if let Ok(s) = stream {
            let _ = s.play();
            while running.load(Ordering::Relaxed) {
                thread::sleep(Duration::from_secs(1));
            }
        }
    });
}

// --- System (CPU / Memory Stats) ---

pub fn start_system_harvester(
    tx: Sender<(String, Vec<u8>)>,
    running: Arc<AtomicBool>,
    state: Arc<Mutex<SharedState>>,
) {
    thread::spawn(move || {
        use sysinfo::{System, RefreshKind, CpuRefreshKind, MemoryRefreshKind};

        // Only load CPU + memory info (NOT processes/disks/networks)
        // System::new_all() enumerates ALL processes on Windows = 2-5 second delay
        // new_with_specifics() with just CPU + memory = instant startup
        let mut sys = System::new_with_specifics(
            RefreshKind::new()
                .with_cpu(CpuRefreshKind::everything())
                .with_memory(MemoryRefreshKind::everything())
        );

        let mut health_tester = NistHealthTester::new();
        health_tester.start();

        // Initial delay - CPU usage needs a baseline measurement (diff-based)
        // sysinfo docs: "you need to call this method at least twice"
        sys.refresh_cpu_usage();
        thread::sleep(Duration::from_millis(300));

        while running.load(Ordering::Relaxed) {
            let enabled = {
                match state.try_lock() {
                    Some(lock) => lock.harvester_states.system,
                    None => false,  // Skip this cycle if lock busy
                }
            };
            if enabled {
                // Targeted refresh: ONLY CPU usage + memory
                sys.refresh_cpu_usage();
                sys.refresh_memory();

                let mut raw_bytes = Vec::with_capacity(256);

                for cpu in sys.cpus() {
                    let usage_bits = cpu.cpu_usage().to_bits();
                    let freq = cpu.frequency();
                    raw_bytes.extend_from_slice(&usage_bits.to_le_bytes());
                    raw_bytes.extend_from_slice(&freq.to_le_bytes());
                }

                let nanos = get_timestamp_nanos();
                raw_bytes.extend_from_slice(&nanos.to_le_bytes());
                raw_bytes.extend_from_slice(&sys.used_memory().to_le_bytes());
                raw_bytes.extend_from_slice(&sys.available_memory().to_le_bytes());

                if raw_bytes.len() > 16 {
                    // Only process if we have real CPU data (not just timestamps)
                    let passed = health_tester.process_batch(&raw_bytes);
                    if !passed.is_empty() {
                        let _ = tx.try_send(("SYSTEM".to_string(), passed));
                    }
                }

                // Use try_lock to avoid blocking other threads
                if let Some(mut lock) = state.try_lock() {
                    let metrics = lock.source_metrics.entry("SYSTEM".to_string()).or_default();
                    metrics.health_state = health_tester.state_name().to_string();
                }
            }
            thread::sleep(Duration::from_millis(500));
        }
    });
}

// --- Mouse / HID (Timing Jitter) ---

/// Mouse / HID Harvester (Lazy-Start, Lock-Free)
///
/// CRITICAL: rdev::listen() installs SetWindowsHookEx(WH_MOUSE_LL) on Windows.
/// This function is NOT called at engine startup - only when user enables mouse.
///
/// The callback MUST return in microseconds. Rules:
///   1. NO Mutex locks in callback (AtomicBool for enable check)
///   2. Skip 98% of events (every 50th)
///   3. try_lock() only on health tester, never block
///   4. try_send() only on channel, never block
pub fn start_mouse_harvester(
    tx: Sender<(String, Vec<u8>)>,
    running: Arc<AtomicBool>,
    state: Arc<Mutex<SharedState>>,
) {
    let mouse_enabled = Arc::new(AtomicBool::new(true));
    let mouse_enabled_hook = mouse_enabled.clone();

    // Poller: syncs AtomicBool from SharedState every 200ms
    let mouse_enabled_poll = mouse_enabled.clone();
    let state_poll = state.clone();
    let running_poll = running.clone();
    thread::spawn(move || {
        while running_poll.load(Ordering::Relaxed) {
            if let Some(lock) = state_poll.try_lock() {
                mouse_enabled_poll.store(lock.harvester_states.mouse, Ordering::Relaxed);
            }
            thread::sleep(Duration::from_millis(200));
        }
    });

    thread::spawn(move || {
        use rdev::{listen, EventType};

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        let last_nanos = Arc::new(AtomicUsize::new(0));
        let last_nanos_clone = last_nanos.clone();
        let health_tester = Arc::new(Mutex::new({
            let mut h = NistHealthTester::new();
            h.start();
            h
        }));
        let ht_clone = health_tester.clone();

        let callback = move |event: rdev::Event| {
            // All checks are atomic - zero mutex operations
            if !running.load(Ordering::Relaxed) { return; }
            if !mouse_enabled_hook.load(Ordering::Relaxed) { return; }

            match event.event_type {
                EventType::MouseMove { x, y } => {
                    let count = counter_clone.fetch_add(1, Ordering::Relaxed);
                    if count % 50 != 0 { return; }  // 98% skip rate

                    let now_nanos = get_timestamp_nanos() as usize;
                    let prev = last_nanos_clone.swap(now_nanos, Ordering::Relaxed);
                    let delta = now_nanos.wrapping_sub(prev) as u64;

                    let mut payload = Vec::with_capacity(24);
                    payload.extend_from_slice(&(x as f64).to_bits().to_le_bytes());
                    payload.extend_from_slice(&(y as f64).to_bits().to_le_bytes());
                    payload.extend_from_slice(&delta.to_le_bytes());

                    if let Some(mut ht) = ht_clone.try_lock() {
                        let passed = ht.process_batch(&payload);
                        if !passed.is_empty() {
                            let _ = tx.try_send(("MOUSE".to_string(), passed));
                        }
                    }
                }
                EventType::ButtonPress(_) => {
                    let now_nanos = get_timestamp_nanos() as usize;
                    let prev = last_nanos_clone.swap(now_nanos, Ordering::Relaxed);
                    let delta = now_nanos.wrapping_sub(prev) as u64;

                    let mut payload = Vec::with_capacity(16);
                    payload.extend_from_slice(&delta.to_le_bytes());
                    payload.extend_from_slice(&(now_nanos as u64).to_le_bytes());

                    if let Some(mut ht) = ht_clone.try_lock() {
                        let passed = ht.process_batch(&payload);
                        if !passed.is_empty() {
                            let _ = tx.try_send(("MOUSE_CLK".to_string(), passed));
                        }
                    }
                }
                _ => {}
            }
        };

        let _ = listen(callback);
    });
}

// --- Video (Camera Noise) ---

pub fn start_video_harvester(
    tx: Sender<(String, Vec<u8>)>,
    running: Arc<AtomicBool>,
    state: Arc<Mutex<SharedState>>,
) {
    thread::spawn(move || {
        use nokhwa::pixel_format::RgbFormat;
        use nokhwa::utils::{CameraIndex, RequestedFormat, RequestedFormatType};
        use nokhwa::Camera;

        // Read camera index from device config
        let cam_idx = state.lock().device_config.camera_device_index;
        let index = CameraIndex::Index(cam_idx as u32);
        let format = RequestedFormat::new::<RgbFormat>(RequestedFormatType::AbsoluteHighestFrameRate);

        if let Ok(mut camera) = Camera::new(index, format) {
            if camera.open_stream().is_ok() {
                let mut last_frame_hash: Option<[u8; 32]> = None;
                let mut health_tester = NistHealthTester::new();
                health_tester.start();

                while running.load(Ordering::Relaxed) {
                    let enabled = state.try_lock()
                        .map(|l| l.harvester_states.video)
                        .unwrap_or(false);
                    if enabled {
                        if let Ok(frame) = camera.frame() {
                            let buffer = frame.buffer();
                            let mut noise: Vec<u8> = buffer.iter()
                                .step_by(7)
                                .map(|&b| b & 0x0F)  // Low 4 bits
                                .take(512)
                                .collect();

                            let nanos = get_timestamp_nanos();
                            noise.extend_from_slice(&nanos.to_le_bytes());

                            // XOR with previous frame hash for diff-based entropy
                            if let Some(ref prev_hash) = last_frame_hash {
                                for (i, b) in noise.iter_mut().enumerate().take(32) {
                                    *b ^= prev_hash[i % 32];
                                }
                            }

                            // Update frame hash
                            let mut hasher = Sha3_256::new();
                            hasher.update(&noise);
                            last_frame_hash = Some(hasher.finalize().into());

                            let passed = health_tester.process_batch(&noise);
                            if !passed.is_empty() {
                                let _ = tx.try_send(("VIDEO".to_string(), passed));
                            }

                            if let Some(mut lock) = state.try_lock() {
                                let metrics = lock.source_metrics.entry("VIDEO".to_string()).or_default();
                                metrics.health_state = health_tester.state_name().to_string();
                            }
                        }
                    }
                    thread::sleep(Duration::from_secs(1));
                }
            }
        }
    });
}

// ============================================================================
// GPU HARVESTERS (Per-GPU Independent Threads)
// CUDA and OpenCL each run their own health tester and thread
// ============================================================================

// --- GPU CUDA (NVIDIA) - Independent Thread ---

pub fn start_gpu_cuda_harvester(
    tx: Sender<(String, Vec<u8>)>,
    running: Arc<AtomicBool>,
    state: Arc<Mutex<SharedState>>,
) {
    thread::spawn(move || {
        let mut health_tester = NistHealthTester::new();
        health_tester.start();

        while running.load(Ordering::Relaxed) {
            let enabled = state.try_lock()
                .map(|l| l.harvester_states.gpu_cuda)
                .unwrap_or(false);
            if enabled {
                #[cfg(feature = "gpu-cuda")]
                {
                    if let Some(gpu_bytes) = cuda_entropy::harvest_cuda(512) {
                        let mut data = gpu_bytes;
                        data.extend_from_slice(&get_timestamp_nanos().to_le_bytes());

                        let passed = health_tester.process_batch(&data);
                        if !passed.is_empty() {
                            let _ = tx.try_send(("GPU_CUDA".to_string(), passed));

                            if let Some(mut lock) = state.try_lock() {
                                let metrics = lock.source_metrics
                                    .entry("GPU_CUDA".to_string()).or_default();
                                metrics.health_state = health_tester.state_name().to_string();
                            }
                        }
                    }
                }
            }
            thread::sleep(Duration::from_millis(500));
        }
    });
}

// --- GPU OpenCL (AMD/Intel) - Independent Thread ---

pub fn start_gpu_ocl_harvester(
    tx: Sender<(String, Vec<u8>)>,
    running: Arc<AtomicBool>,
    state: Arc<Mutex<SharedState>>,
) {
    thread::spawn(move || {
        let mut health_tester = NistHealthTester::new();
        health_tester.start();

        while running.load(Ordering::Relaxed) {
            let enabled = state.try_lock()
                .map(|l| l.harvester_states.gpu_ocl)
                .unwrap_or(false);
            if enabled {
                #[cfg(feature = "gpu-opencl")]
                {
                    if let Some(gpu_bytes) = opencl_entropy::harvest_opencl(512) {
                        let mut data = gpu_bytes;
                        data.extend_from_slice(&get_timestamp_nanos().to_le_bytes());

                        let passed = health_tester.process_batch(&data);
                        if !passed.is_empty() {
                            let _ = tx.try_send(("GPU_OCL".to_string(), passed));

                            if let Some(mut lock) = state.try_lock() {
                                let metrics = lock.source_metrics
                                    .entry("GPU_OCL".to_string()).or_default();
                                metrics.health_state = health_tester.state_name().to_string();
                            }
                        }
                    }
                }
            }
            thread::sleep(Duration::from_millis(500));
        }
    });
}

// ============================================================================
// GUITAR ESP32 UDP ENTROPY LISTENER
// Listens on 6 UDP ports (3 data + 3 ctrl) for guitar entropy
// Spectra:   5005/5056
// Neptonius: 5006/5057
// Thalyn:    5007/5058
// ============================================================================

pub fn start_guitar_udp_listener(
    tx: Sender<(String, Vec<u8>)>,
    running: Arc<AtomicBool>,
    state: Arc<Mutex<SharedState>>,
) {
    // Collect guitar port configs before spawning threads
    // NOTE: This runs once at startup, blocking lock is fine here
    let guitar_configs: Vec<(String, u16, u16)> = {
        let lock = state.lock();
        lock.guitar_states.iter().map(|(name, gs)| {
            (name.clone(), gs.data_port, gs.ctrl_port)
        }).collect()
    };

    // Spawn one listener thread per guitar (handles both data + ctrl ports)
    for (gname, data_port, ctrl_port) in guitar_configs {
        let tx_data = tx.clone();
        let tx_ctrl = tx.clone();
        let running_data = running.clone();
        let running_ctrl = running.clone();
        let state_data = state.clone();
        let state_ctrl = state.clone();
        let gname_data = gname.clone();
        let gname_ctrl = gname.clone();

        // Data port listener
        thread::spawn(move || {
            use std::net::UdpSocket;

            let bind_addr = format!("0.0.0.0:{}", data_port);
            let socket = match UdpSocket::bind(&bind_addr) {
                Ok(s) => {
                    // Startup log - blocking lock is fine (runs once)
                    if let Some(mut lock) = state_data.try_lock() {
                        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
                        let msg = format!("[{}] GUITAR {}: Listening on UDP:{} (data)",
                            ts, gname_data, data_port);
                        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
                        lock.logs.push_back(msg);
                    }
                    s
                }
                Err(e) => {
                    eprintln!("GUITAR {}: Failed to bind UDP:{} - {}", gname_data, data_port, e);
                    return;
                }
            };

            socket.set_read_timeout(Some(Duration::from_millis(500))).ok();

            let mut health_tester = NistHealthTester::new();
            health_tester.start();

            let mut buf = [0u8; 2048];

            while running_data.load(Ordering::Relaxed) {
                // Check if this guitar is enabled (try_lock - never block hot path)
                let enabled = state_data.try_lock()
                    .and_then(|l| l.guitar_states.get(&gname_data).map(|gs| gs.enabled))
                    .unwrap_or(false);

                if !enabled {
                    thread::sleep(Duration::from_millis(100));
                    continue;
                }

                match socket.recv_from(&mut buf) {
                    Ok((len, _addr)) => {
                        if len == 0 { continue; }

                        let mut data = buf[..len].to_vec();
                        data.extend_from_slice(&get_timestamp_nanos().to_le_bytes());

                        let passed = health_tester.process_batch(&data);
                        if !passed.is_empty() {
                            let source = format!("GUITAR_{}", gname_data.to_uppercase());
                            let _ = tx_data.try_send((source, passed));
                        }

                        // Update guitar stats (try_lock - skip if busy)
                        if let Some(mut lock) = state_data.try_lock() {
                            if let Some(gs) = lock.guitar_states.get_mut(&gname_data) {
                                gs.packets_received += 1;
                                gs.bytes_received += len as u64;
                            }
                            let metrics = lock.source_metrics
                                .entry(format!("GUITAR_{}", gname_data.to_uppercase()))
                                .or_default();
                            metrics.health_state = health_tester.state_name().to_string();
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                        // Timeout - normal for non-blocking
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {
                        // Timeout - normal
                    }
                    Err(_) => {
                        thread::sleep(Duration::from_millis(100));
                    }
                }
            }
        });

        // Control port listener (same pattern, separate thread)
        thread::spawn(move || {
            use std::net::UdpSocket;

            let bind_addr = format!("0.0.0.0:{}", ctrl_port);
            let socket = match UdpSocket::bind(&bind_addr) {
                Ok(s) => {
                    if let Some(mut lock) = state_ctrl.try_lock() {
                        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
                        let msg = format!("[{}] GUITAR {}: Listening on UDP:{} (ctrl)",
                            ts, gname_ctrl, ctrl_port);
                        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
                        lock.logs.push_back(msg);
                    }
                    s
                }
                Err(e) => {
                    eprintln!("GUITAR {}: Failed to bind UDP:{} - {}", gname_ctrl, ctrl_port, e);
                    return;
                }
            };

            socket.set_read_timeout(Some(Duration::from_millis(500))).ok();

            let mut health_tester = NistHealthTester::new();
            health_tester.start();

            let mut buf = [0u8; 2048];

            while running_ctrl.load(Ordering::Relaxed) {
                let enabled = state_ctrl.try_lock()
                    .and_then(|l| l.guitar_states.get(&gname_ctrl).map(|gs| gs.enabled))
                    .unwrap_or(false);

                if !enabled {
                    thread::sleep(Duration::from_millis(100));
                    continue;
                }

                match socket.recv_from(&mut buf) {
                    Ok((len, _addr)) => {
                        if len == 0 { continue; }

                        let mut data = buf[..len].to_vec();
                        data.extend_from_slice(&get_timestamp_nanos().to_le_bytes());

                        let passed = health_tester.process_batch(&data);
                        if !passed.is_empty() {
                            let source = format!("GUITAR_{}_CTRL", gname_ctrl.to_uppercase());
                            let _ = tx_ctrl.try_send((source, passed));
                        }

                        // Update guitar stats (ctrl packets also count)
                        if let Some(mut lock) = state_ctrl.try_lock() {
                            if let Some(gs) = lock.guitar_states.get_mut(&gname_ctrl) {
                                gs.packets_received += 1;
                                gs.bytes_received += len as u64;
                            }
                        }
                    }
                    Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {}
                    Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {}
                    Err(_) => {
                        thread::sleep(Duration::from_millis(100));
                    }
                }
            }
        });
    }
}

// ============================================================================
// WIFI NOISE HARVESTER
// Reads signal strength, channel noise, and BSSID data as entropy
// Uses system commands or /proc/net/wireless on Linux
// ============================================================================

pub fn start_wifi_harvester(
    tx: Sender<(String, Vec<u8>)>,
    running: Arc<AtomicBool>,
    state: Arc<Mutex<SharedState>>,
) {
    thread::spawn(move || {
        let mut health_tester = NistHealthTester::new();
        health_tester.start();

        while running.load(Ordering::Relaxed) {
            let enabled = state.try_lock()
                .map(|l| l.harvester_states.wifi)
                .unwrap_or(false);
            if enabled {
                let mut noise_data = Vec::with_capacity(256);

                // Strategy 1: Read /proc/net/wireless (Linux)
                if let Ok(contents) = std::fs::read_to_string("/proc/net/wireless") {
                    noise_data.extend_from_slice(contents.as_bytes());
                }

                // Strategy 2: Read signal levels from iwconfig-style info
                // Try to read from /sys/class/net/*/wireless/
                let iface = state.try_lock()
                    .map(|l| l.device_config.wifi_interface.clone())
                    .unwrap_or_default();

                // If interface specified, read its stats
                if !iface.is_empty() {
                    let path = format!("/sys/class/net/{}/statistics/rx_bytes", iface);
                    if let Ok(val) = std::fs::read_to_string(&path) {
                        noise_data.extend_from_slice(val.trim().as_bytes());
                    }
                    let path = format!("/sys/class/net/{}/statistics/tx_bytes", iface);
                    if let Ok(val) = std::fs::read_to_string(&path) {
                        noise_data.extend_from_slice(val.trim().as_bytes());
                    }
                    let path = format!("/sys/class/net/{}/statistics/rx_dropped", iface);
                    if let Ok(val) = std::fs::read_to_string(&path) {
                        noise_data.extend_from_slice(val.trim().as_bytes());
                    }
                } else {
                    // Auto-detect: try common interface names
                    for candidate in &["wlan0", "wlp2s0", "wlp3s0", "wifi0"] {
                        let path = format!("/sys/class/net/{}/statistics/rx_bytes", candidate);
                        if let Ok(val) = std::fs::read_to_string(&path) {
                            noise_data.extend_from_slice(val.trim().as_bytes());
                            // Also grab noise floor if available
                            let noise_path = format!("/sys/class/net/{}/statistics/collisions", candidate);
                            if let Ok(nval) = std::fs::read_to_string(&noise_path) {
                                noise_data.extend_from_slice(nval.trim().as_bytes());
                            }
                            break;
                        }
                    }
                }

                // Add timestamp for additional jitter
                noise_data.extend_from_slice(&get_timestamp_nanos().to_le_bytes());

                // On Windows, fallback to timestamp + process info
                #[cfg(target_os = "windows")]
                {
                    // Use netsh output or WMI if available
                    if let Ok(output) = std::process::Command::new("netsh")
                        .args(&["wlan", "show", "interfaces"])
                        .output()
                    {
                        noise_data.extend_from_slice(&output.stdout);
                    }
                }

                if noise_data.len() > 8 {
                    let passed = health_tester.process_batch(&noise_data);
                    if !passed.is_empty() {
                        let _ = tx.try_send(("WIFI".to_string(), passed));

                        if let Some(mut lock) = state.try_lock() {
                            lock.wifi_active = true;
                            lock.wifi_samples += 1;
                            let metrics = lock.source_metrics
                                .entry("WIFI".to_string()).or_default();
                            metrics.health_state = health_tester.state_name().to_string();
                        }
                    }
                }
            } else {
                if let Some(mut lock) = state.try_lock() {
                    lock.wifi_active = false;
                }
            }
            thread::sleep(Duration::from_secs(2));
        }
    });
}

// ============================================================================
// USB SERIAL HARVESTER
// Reads entropy data from USB serial devices (ESP32, Arduino, etc.)
// Uses the serialport crate for cross-platform support
// ============================================================================

pub fn start_usb_serial_harvester(
    tx: Sender<(String, Vec<u8>)>,
    running: Arc<AtomicBool>,
    state: Arc<Mutex<SharedState>>,
) {
    thread::spawn(move || {
        let mut health_tester = NistHealthTester::new();
        health_tester.start();

        let mut current_port: Option<Box<dyn serialport::SerialPort>> = None;
        let mut last_port_name = String::new();

        while running.load(Ordering::Relaxed) {
            let enabled = state.try_lock()
                .map(|l| l.harvester_states.usb_serial)
                .unwrap_or(false);
            if !enabled {
                current_port = None;
                if let Some(mut lock) = state.try_lock() {
                    lock.usb_serial_active = false;
                }
                thread::sleep(Duration::from_millis(500));
                continue;
            }

            // Check if port config changed
            let (port_name, baud) = state.try_lock()
                .map(|l| (l.device_config.usb_serial_port.clone(),
                           l.device_config.usb_serial_baud))
                .unwrap_or_else(|| (String::new(), 115200));

            // Auto-detect if no port specified
            let target_port = if port_name.is_empty() {
                // Try to find any available serial port
                match serialport::available_ports() {
                    Ok(ports) => {
                        ports.into_iter()
                            .next()
                            .map(|p| p.port_name)
                            .unwrap_or_default()
                    }
                    Err(_) => String::new(),
                }
            } else {
                port_name
            };

            if target_port.is_empty() {
                if let Some(mut lock) = state.try_lock() {
                    lock.usb_serial_active = false;
                }
                thread::sleep(Duration::from_secs(2));
                continue;
            }

            // Reconnect if port changed
            if target_port != last_port_name || current_port.is_none() {
                current_port = serialport::new(&target_port, baud)
                    .timeout(Duration::from_millis(500))
                    .open()
                    .ok();

                if current_port.is_some() {
                    last_port_name = target_port.clone();
                    if let Some(mut lock) = state.try_lock() {
                        lock.usb_serial_active = true;
                        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
                        let msg = format!("[{}] USB: Opened {} @ {}", ts, target_port, baud);
                        if lock.logs.len() >= 500 { lock.logs.pop_front(); }
                        lock.logs.push_back(msg);
                    }
                } else {
                    if let Some(mut lock) = state.try_lock() {
                        lock.usb_serial_active = false;
                    }
                    thread::sleep(Duration::from_secs(2));
                    continue;
                }
            }

            // Read data from serial port
            if let Some(ref mut port) = current_port {
                let mut buf = [0u8; 512];
                match port.read(&mut buf) {
                    Ok(len) if len > 0 => {
                        let mut data = buf[..len].to_vec();
                        data.extend_from_slice(&get_timestamp_nanos().to_le_bytes());

                        let passed = health_tester.process_batch(&data);
                        if !passed.is_empty() {
                            let _ = tx.try_send(("USB_SERIAL".to_string(), passed));

                            if let Some(mut lock) = state.try_lock() {
                                lock.usb_serial_bytes += len as u64;
                                let metrics = lock.source_metrics
                                    .entry("USB_SERIAL".to_string()).or_default();
                                metrics.health_state = health_tester.state_name().to_string();
                            }
                        }
                    }
                    Ok(_) => {}
                    Err(ref e) if e.kind() == std::io::ErrorKind::TimedOut => {}
                    Err(_) => {
                        // Port disconnected
                        current_port = None;
                        if let Some(mut lock) = state.try_lock() {
                            lock.usb_serial_active = false;
                        }
                    }
                }
            }

            thread::sleep(Duration::from_millis(500));
        }
    });
}

// ============================================================================
// P2P SERVER WITH HMAC AUTHENTICATION
// ============================================================================

fn validate_hmac(
    key: &[u8],
    node_id: &str,
    seq: u64,
    timestamp: i64,
    payload: &[u8],
    sources: &str,
    claimed_mac: &str,
) -> bool {
    let mut mac = match HmacSha256::new_from_slice(key) {
        Ok(m) => m,
        Err(_) => return false,
    };

    mac.update(node_id.as_bytes());
    mac.update(b"|");
    mac.update(&seq.to_le_bytes());
    mac.update(b"|");
    mac.update(&timestamp.to_le_bytes());
    mac.update(b"|");
    mac.update(sources.as_bytes());
    mac.update(b"|");
    mac.update(payload);

    // SECURITY: Use constant-time comparison to prevent timing side-channel attacks.
    // hmac::verify_slice uses subtle::ConstantTimeEq internally.
    let claimed_bytes = match hex::decode(claimed_mac) {
        Ok(b) => b,
        Err(_) => return false,
    };
    mac.verify_slice(&claimed_bytes).is_ok()
}

pub fn start_p2p_server(
    tx: Sender<(String, Vec<u8>)>,
    state: Arc<Mutex<SharedState>>,
    running: Arc<AtomicBool>,
) {
    thread::spawn(move || {
        use std::net::TcpListener;
        use std::io::{Read, Write};

        // Startup config read - blocking lock OK (runs once)
        let port = state.lock().p2p_config.listen_port;
        let addr = format!("0.0.0.0:{}", port);

        let listener = match TcpListener::bind(&addr) {
            Ok(l) => {
                let ts = chrono::Local::now().format("%H:%M:%S").to_string();
                let mut lock = state.lock();
                let hmac_status = if lock.p2p_config.hmac_key.is_some() {
                    "ENABLED"
                } else {
                    "DISABLED"
                };
                let msg = format!("[{}] P2P: Listening on port {} (HMAC: {})",
                    ts, port, hmac_status);
                if lock.logs.len() >= 500 { lock.logs.pop_front(); }
                lock.logs.push_back(msg);
                drop(lock);
                l
            }
            Err(e) => {
                eprintln!("P2P: Failed to bind to {}: {}", addr, e);
                return;
            }
        };

        listener.set_nonblocking(true).ok();

        let mut health_tester = NistHealthTester::new();
        health_tester.start();

        while running.load(Ordering::Relaxed) {
            match listener.accept() {
                Ok((mut stream, addr)) => {
                    let p2p_active = state.try_lock()
                        .map(|l| l.p2p_config.active)
                        .unwrap_or(false);
                    if !p2p_active {
                        continue;
                    }

                    let tx_clone = tx.clone();
                    let state_clone = state.clone();

                    thread::spawn(move || {
                        let mut buffer = String::new();
                        if stream.read_to_string(&mut buffer).is_ok() {
                            if let Some(body_start) = buffer.find("\r\n\r\n") {
                                let body = &buffer[body_start + 4..];

                                if let Ok(json) = serde_json::from_str::<serde_json::Value>(body) {
                                    let payload_hex = json["payload_hex"].as_str().unwrap_or("");
                                    let entropy_bytes = match hex::decode(payload_hex) {
                                        Ok(b) => b,
                                        Err(_) => {
                                            let resp = "HTTP/1.1 400 Bad Request\r\nContent-Length: 11\r\n\r\nINVALID_HEX";
                                            let _ = stream.write_all(resp.as_bytes());
                                            return;
                                        }
                                    };

                                    // Check HMAC if configured (try_lock on hot path)
                                    if let Some(lock) = state_clone.try_lock() {
                                        if let Some(ref hmac_key) = lock.p2p_config.hmac_key {
                                            let node_id = json["node_id"].as_str().unwrap_or("");
                                            let seq = json["seq"].as_u64().unwrap_or(0);
                                            let timestamp = json["timestamp"].as_i64().unwrap_or(0);
                                            let sources = json["sources"].as_str().unwrap_or("");
                                            let mac_hex = json["mac_hex"].as_str().unwrap_or("");

                                            if !validate_hmac(hmac_key, node_id, seq, timestamp,
                                                &entropy_bytes, sources, mac_hex)
                                            {
                                                drop(lock);
                                                let resp = "HTTP/1.1 403 Forbidden\r\nContent-Length: 12\r\n\r\nHMAC_INVALID";
                                                let _ = stream.write_all(resp.as_bytes());
                                                return;
                                            }

                                            // Check timestamp (within 5 minutes)
                                            let now = get_timestamp() as i64;
                                            if (now - timestamp).abs() > 300 {
                                                drop(lock);
                                                let resp = "HTTP/1.1 403 Forbidden\r\nContent-Length: 13\r\n\r\nTIME_MISMATCH";
                                                let _ = stream.write_all(resp.as_bytes());
                                                return;
                                            }
                                        }
                                        drop(lock);
                                    } else {
                                        // Lock busy - reject gracefully rather than block
                                        let resp = "HTTP/1.1 503 Service Unavailable\r\nContent-Length: 4\r\n\r\nBUSY";
                                        let _ = stream.write_all(resp.as_bytes());
                                        return;
                                    }

                                    // Add to processing queue
                                    let source = format!("P2P_{}", addr.ip());
                                    let _ = tx_clone.try_send((source, entropy_bytes));

                                    // Update P2P stats (try_lock)
                                    if let Some(mut lock) = state_clone.try_lock() {
                                        lock.p2p_config.received_count += 1;
                                    }

                                    let resp = "HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK";
                                    let _ = stream.write_all(resp.as_bytes());
                                    return;
                                }
                            }
                        }

                        let resp = "HTTP/1.1 400 Bad Request\r\nContent-Length: 5\r\n\r\nERROR";
                        let _ = stream.write_all(resp.as_bytes());
                    });
                }
                Err(ref e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                    thread::sleep(Duration::from_millis(100));
                }
                Err(_) => {
                    thread::sleep(Duration::from_millis(100));
                }
            }
        }
    });
}

// ============================================================================
// HEADSCALE FORWARDER (Aoi Midori)
// Periodic thread that monitors headscale connectivity
// Actual forwarding happens in the mixer thread (lib.rs)
// This thread handles health checks and reconnection
// ============================================================================

pub fn start_headscale_forwarder(
    state: Arc<Mutex<SharedState>>,
    running: Arc<AtomicBool>,
) {
    thread::spawn(move || {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_millis(2000))
            .build()
            .unwrap_or_else(|_| reqwest::blocking::Client::new());

        // Log startup (blocking lock OK - runs once)
        {
            let mut lock = state.lock();
            let ts = chrono::Local::now().format("%H:%M:%S").to_string();
            let msg = format!("[{}] HEADSCALE: Forwarder ready -> {}:{}",
                ts, lock.headscale.target_ip, lock.headscale.target_port);
            if lock.logs.len() >= 500 { lock.logs.pop_front(); }
            lock.logs.push_back(msg);
        }

        let mut last_ping = Instant::now();

        while running.load(Ordering::Relaxed) {
            let enabled = state.try_lock()
                .map(|l| l.headscale.enabled)
                .unwrap_or(false);

            if enabled && last_ping.elapsed() >= Duration::from_secs(30) {
                // Periodic health ping to Aoi Midori
                let (url, ip) = match state.try_lock() {
                    Some(lock) => {
                        let url = format!("http://{}:{}/ping",
                            lock.headscale.target_ip, lock.headscale.target_port);
                        let ip = lock.headscale.target_ip.clone();
                        (url, ip)
                    }
                    None => {
                        thread::sleep(Duration::from_secs(5));
                        continue;
                    }
                };

                match client.get(&url).send() {
                    Ok(resp) if resp.status().is_success() => {
                        if let Some(mut lock) = state.try_lock() {
                            let ts = chrono::Local::now().format("%H:%M:%S").to_string();
                            let msg = format!("[{}] HEADSCALE: Aoi Midori reachable ({})",
                                ts, ip);
                            if lock.logs.len() >= 500 { lock.logs.pop_front(); }
                            lock.logs.push_back(msg);
                        }
                    }
                    _ => {
                        // Ping failed - not critical, forwarding will retry
                    }
                }

                last_ping = Instant::now();
            }

            thread::sleep(Duration::from_secs(5));
        }
    });
}
