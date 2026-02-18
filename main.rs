//! NullMagnet Live - main.rs (Tauri Entry Point)
//! Jupiter Labs
//!
//! This file replaces main.py + config.py for the Tauri desktop app.
//! Every #[tauri::command] maps to a Python callback from main.py.
//!
//! ChaosEngine is constructed via new_native() from lib.rs.
//! All methods are native Rust (no PyO3).

#![cfg_attr(
    all(not(debug_assertions), target_os = "windows"),
    windows_subsystem = "windows"
)]

use tauri::State;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;

// Import the core engine (from lib.rs - native Rust, no PyO3)
use nullmagnet_live_core::ChaosEngine;

// ============================================================================
// APP STATE
// Wraps ChaosEngine so Tauri commands can access it via State<>
// ============================================================================

struct AppState {
    engine: ChaosEngine,
}

// ============================================================================
// CONFIG (replaces config.py)
// ============================================================================

const CONFIG_FILE: &str = "nullmagnet_config.json";

#[derive(Serialize, Deserialize, Clone)]
struct NullMagnetConfig {
    version: String,
    branding: String,
    audio_device_index: Option<usize>,
    audio_gain: f64,
    audio_sample_rate: u32,
    camera_device_indices: Vec<usize>,
    camera_resolution: (u32, u32),
    usb_serial_ports: Vec<String>,
    usb_serial_baud: u32,
    wifi_interface: String,
    guitar_config: HashMap<String, GuitarConfigEntry>,
    headscale_enabled: bool,
    headscale_target_ip: String,
    headscale_target_port: u16,
    live_mode_default: bool,
    uplink_ip: String,
    uplink_port: u16,
    p2p_port: u16,
    mint_threshold_bits: u32,
    gpu_cuda_device_id: usize,
    gpu_ocl_platform_id: usize,
    gpu_ocl_device_id: usize,
}

#[derive(Serialize, Deserialize, Clone)]
struct GuitarConfigEntry {
    enabled: bool,
}

impl Default for NullMagnetConfig {
    fn default() -> Self {
        Self {
            version: "1.0.0".into(),
            branding: "NullMagnet Live".into(),
            audio_device_index: None,
            audio_gain: 1.0,
            audio_sample_rate: 44100,
            camera_device_indices: vec![0],
            camera_resolution: (320, 240),
            usb_serial_ports: vec![],
            usb_serial_baud: 115200,
            wifi_interface: String::new(),
            guitar_config: HashMap::from([
                ("Spectra".into(), GuitarConfigEntry { enabled: true }),
                ("Neptonius".into(), GuitarConfigEntry { enabled: true }),
                ("Thalyn".into(), GuitarConfigEntry { enabled: true }),
            ]),
            headscale_enabled: true,
            headscale_target_ip: std::env::var("NULL_HEADSCALE_IP")
                .unwrap_or_else(|_| "100.64.0.15".into()),
            headscale_target_port: std::env::var("NULL_HEADSCALE_PORT")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(8100),
            live_mode_default: false,
            uplink_ip: std::env::var("NULL_UPLINK_IP")
                .unwrap_or_else(|_| "192.168.1.19".into()),
            uplink_port: std::env::var("NULL_UPLINK_PORT")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(8000),
            p2p_port: std::env::var("NULL_P2P_PORT")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(9000),
            mint_threshold_bits: std::env::var("NULL_MINT_THRESHOLD_BITS")
                .ok().and_then(|s| s.parse().ok()).unwrap_or(256),
            gpu_cuda_device_id: 0,
            gpu_ocl_platform_id: 0,
            gpu_ocl_device_id: 0,
        }
    }
}

// ============================================================================
// TAURI COMMANDS
// Each maps to a Python callback in main.py
// ============================================================================

// --- Harvester Toggles (main.py: toggle_harvester) ---

#[tauri::command]
fn toggle_harvester(state: State<AppState>, source: String, enabled: bool) -> Result<String, String> {
    state.engine.toggle_harvester_internal(&source, enabled);
    let status = if enabled { "Active" } else { "Inactive" };
    Ok(format!("{} -> {}", source, status))
}

// --- Guitar Toggles (main.py: toggle_guitar) ---

#[tauri::command]
fn toggle_guitar(state: State<AppState>, name: String, enabled: bool) -> Result<String, String> {
    let rust_name = format!("GUITAR_{}", name.to_uppercase());
    state.engine.toggle_harvester_internal(&rust_name, enabled);

    // Update guitar state
    let mut lock = state.engine.state.lock();
    if let Some(gs) = lock.guitar_states.get_mut(&name) {
        gs.enabled = enabled;
    }

    let status = if enabled { "Active" } else { "Inactive" };
    Ok(format!("Guitar {} -> {}", name, status))
}

// --- Network Toggle (main.py: toggle_network) ---

#[tauri::command]
fn toggle_uplink(state: State<AppState>, enabled: bool) -> Result<String, String> {
    let mut lock = state.engine.state.lock();
    lock.net_mode = enabled;
    let status = if enabled { "ENABLED" } else { "DISABLED" };
    Ok(format!("Uplink -> {}", status))
}

// --- Set Network Target IP (main.py: callback_ip_update) ---

#[tauri::command]
fn set_network_target(state: State<AppState>, ip: String) -> Result<String, String> {
    if !ip.is_empty() {
        let mut lock = state.engine.state.lock();
        lock.uplink_url = format!("http://{}:8000/entropy", ip);
        Ok(format!("Target set to {}", ip))
    } else {
        Err("Empty IP".into())
    }
}

// --- Set Uplink Target IP + Port (NEW: from GUI port input) ---

#[tauri::command]
fn set_uplink_target(state: State<AppState>, ip: String, port: u16) -> Result<String, String> {
    if ip.is_empty() {
        return Err("Empty IP".into());
    }
    let port = if port == 0 { 8000 } else { port };
    let mut lock = state.engine.state.lock();
    lock.uplink_url = format!("http://{}:{}/entropy", ip, port);
    let ts = chrono::Local::now().format("%H:%M:%S").to_string();
    let msg = format!("[{}] NET: Uplink target set to {}:{}", ts, ip, port);
    if lock.logs.len() >= 500 { lock.logs.pop_front(); }
    lock.logs.push_back(msg);
    Ok(format!("Uplink -> {}:{}", ip, port))
}

// --- Headscale Toggle (main.py: toggle_headscale) ---

#[tauri::command]
fn toggle_headscale(state: State<AppState>, enabled: bool) -> Result<String, String> {
    let mut lock = state.engine.state.lock();
    lock.headscale.enabled = enabled;
    let status = if enabled { "ENABLED" } else { "DISABLED" };
    let ip = lock.headscale.target_ip.clone();
    Ok(format!("Headscale -> {} (Aoi Midori: {})", status, ip))
}

// --- Set Headscale Target IP + Port (NEW: from GUI inputs) ---

#[tauri::command]
fn set_headscale_target(state: State<AppState>, ip: String, port: u16) -> Result<String, String> {
    if ip.is_empty() {
        return Err("Empty IP".into());
    }
    let port = if port == 0 { 8100 } else { port };
    let mut lock = state.engine.state.lock();
    lock.headscale.target_ip = ip.clone();
    lock.headscale.target_port = port;
    let ts = chrono::Local::now().format("%H:%M:%S").to_string();
    let msg = format!("[{}] HEADSCALE: Target set to Aoi Midori @ {}:{}", ts, ip, port);
    if lock.logs.len() >= 500 { lock.logs.pop_front(); }
    lock.logs.push_back(msg);
    Ok(format!("Headscale -> {}:{}", ip, port))
}

// --- P2P Controls (main.py: toggle_p2p, callback_p2p_port_update, callback_add_peer) ---

#[tauri::command]
fn toggle_p2p(state: State<AppState>, enabled: bool) -> Result<String, String> {
    let mut lock = state.engine.state.lock();
    lock.p2p_config.active = enabled;
    let status = if enabled { "ENABLED" } else { "DISABLED" };
    Ok(format!("P2P -> {}", status))
}

#[tauri::command]
fn set_p2p_port(state: State<AppState>, port: u16) -> Result<String, String> {
    if port >= 1024 {
        let mut lock = state.engine.state.lock();
        lock.p2p_config.listen_port = port;
        Ok(format!("P2P port set to {}", port))
    } else {
        Err("Port must be >= 1024".into())
    }
}

#[tauri::command]
fn add_peer(state: State<AppState>, addr: String) -> Result<String, String> {
    if !addr.is_empty() {
        let mut lock = state.engine.state.lock();
        lock.p2p_config.peers.push(addr.clone());
        Ok(format!("Added peer {}", addr))
    } else {
        Err("Empty address".into())
    }
}

// --- HMAC Key (main.py: callback_set_hmac_key, callback_load_hmac_from_env) ---

#[tauri::command]
fn set_p2p_hmac_key(state: State<AppState>, hex_key: String) -> Result<String, String> {
    match hex::decode(&hex_key) {
        Ok(key_bytes) if key_bytes.len() == 32 => {
            let mut lock = state.engine.state.lock();
            lock.p2p_config.hmac_key = Some(key_bytes);
            Ok("HMAC: ACTIVE".into())
        }
        Ok(key_bytes) => Err(format!("Key must be 32 bytes, got {}", key_bytes.len())),
        Err(e) => Err(format!("Invalid hex: {}", e)),
    }
}

#[tauri::command]
fn load_hmac_from_env() -> Result<String, String> {
    match std::env::var("NULL_P2P_HMAC_KEY_HEX") {
        Ok(key_hex) if !key_hex.is_empty() => Ok(key_hex),
        _ => Err("No HMAC key in environment".into()),
    }
}

// --- PQC Minting (main.py: callback_mint_pqc) ---

#[tauri::command]
fn mint_pqc_bundle(state: State<AppState>, requester: String) -> Result<String, String> {
    // Delegate to engine's mint method
    state.engine.mint_pqc_bundle_internal(&requester)
        .map_err(|e| format!("Mint error: {}", e))
}

// --- Auto-Mint Toggle (main.py: callback_toggle_auto_mint) ---

#[tauri::command]
fn set_auto_mint(state: State<AppState>, enabled: bool) -> Result<String, String> {
    let mut lock = state.engine.state.lock();
    lock.auto_mint_enabled = enabled;
    let status = if enabled { "ENABLED" } else { "DISABLED" };
    Ok(format!("Auto-mint -> {}", status))
}

// --- Device Enumeration (NEW: populate dropdowns in GUI) ---

#[tauri::command]
fn list_audio_devices() -> Result<String, String> {
    use cpal::traits::{HostTrait, DeviceTrait};
    let host = cpal::default_host();
    let mut devices = Vec::new();

    // List input devices (microphones) for entropy harvesting
    if let Ok(input_devices) = host.input_devices() {
        for (i, device) in input_devices.enumerate() {
            let name = device.name().unwrap_or_else(|_| format!("Audio Input #{}", i));
            let has_input = device.default_input_config().is_ok();
            if has_input {
                devices.push(serde_json::json!({
                    "index": i,
                    "name": name,
                    "type": "input",
                }));
            }
        }
    }

    // If no input devices found, add a placeholder
    if devices.is_empty() {
        devices.push(serde_json::json!({
            "index": 0,
            "name": "(No audio input devices found)",
            "type": "none",
        }));
    }

    Ok(serde_json::json!(devices).to_string())
}

#[tauri::command]
fn list_camera_devices() -> Result<String, String> {
    let mut devices = Vec::new();

    // nokhwa::query() enumerates cameras via platform backend (Media Foundation on Windows, V4L2 on Linux)
    match nokhwa::query(nokhwa::utils::ApiBackend::Auto) {
        Ok(camera_list) => {
            for info in &camera_list {
                let idx = match info.index() {
                    nokhwa::utils::CameraIndex::Index(i) => *i as usize,
                    nokhwa::utils::CameraIndex::String(s) => {
                        // Try to parse string index, fallback to 0
                        s.parse::<usize>().unwrap_or(0)
                    }
                };
                devices.push(serde_json::json!({
                    "index": idx,
                    "name": info.human_name(),
                    "description": info.description(),
                    "misc": info.misc(),
                }));
            }
        }
        Err(e) => {
            // Camera enumeration failed - log it but don't error out
            devices.push(serde_json::json!({
                "index": 0,
                "name": format!("Camera query failed: {}", e),
                "description": "",
                "misc": "",
            }));
        }
    }

    if devices.is_empty() {
        devices.push(serde_json::json!({
            "index": 0,
            "name": "(No cameras found)",
            "description": "",
            "misc": "",
        }));
    }

    Ok(serde_json::json!(devices).to_string())
}

#[tauri::command]
fn list_serial_ports() -> Result<String, String> {
    let mut ports = Vec::new();

    match serialport::available_ports() {
        Ok(port_list) => {
            for port in &port_list {
                let port_type = match &port.port_type {
                    serialport::SerialPortType::UsbPort(usb) => {
                        let mfg = usb.manufacturer.as_deref().unwrap_or("Unknown");
                        let product = usb.product.as_deref().unwrap_or("USB Serial");
                        format!("{} - {}", product, mfg)
                    }
                    serialport::SerialPortType::BluetoothPort => "Bluetooth Serial".to_string(),
                    serialport::SerialPortType::PciPort => "PCI Serial".to_string(),
                    serialport::SerialPortType::Unknown => "Serial Port".to_string(),
                };
                ports.push(serde_json::json!({
                    "port": port.port_name,
                    "description": port_type,
                }));
            }
        }
        Err(e) => {
            ports.push(serde_json::json!({
                "port": "",
                "description": format!("Port enumeration failed: {}", e),
            }));
        }
    }

    if ports.is_empty() {
        ports.push(serde_json::json!({
            "port": "",
            "description": "(No serial ports found)",
        }));
    }

    Ok(serde_json::json!(ports).to_string())
}

// --- Device Manager (main.py: callback_audio_device, _gain, _camera, _usb, _wifi) ---

#[tauri::command]
fn set_audio_device(state: State<AppState>, index: usize) -> Result<String, String> {
    let mut lock = state.engine.state.lock();
    lock.device_config.audio_device_index = Some(index);
    let ts = chrono::Local::now().format("%H:%M:%S").to_string();
    let msg = format!("[{}] DEVICE: Audio device set to index {}", ts, index);
    if lock.logs.len() >= 500 { lock.logs.pop_front(); }
    lock.logs.push_back(msg);
    Ok(format!("Audio device -> {}", index))
}

#[tauri::command]
fn set_audio_gain(state: State<AppState>, gain: f64) -> Result<String, String> {
    let mut lock = state.engine.state.lock();
    lock.device_config.audio_gain = gain.max(0.1).min(10.0);
    Ok(format!("Audio gain -> {:.1}", lock.device_config.audio_gain))
}

#[tauri::command]
fn set_camera_device(state: State<AppState>, index: usize) -> Result<String, String> {
    let mut lock = state.engine.state.lock();
    lock.device_config.camera_device_index = index;
    let ts = chrono::Local::now().format("%H:%M:%S").to_string();
    let msg = format!("[{}] DEVICE: Camera set to index {}", ts, index);
    if lock.logs.len() >= 500 { lock.logs.pop_front(); }
    lock.logs.push_back(msg);
    Ok(format!("Camera -> {}", index))
}

#[tauri::command]
fn set_usb_serial_port(state: State<AppState>, port: String, baud: u32) -> Result<String, String> {
    let mut lock = state.engine.state.lock();
    lock.device_config.usb_serial_port = port.clone();
    lock.device_config.usb_serial_baud = baud;
    let ts = chrono::Local::now().format("%H:%M:%S").to_string();
    let msg = format!("[{}] DEVICE: USB serial set to {} @ {}", ts, port, baud);
    if lock.logs.len() >= 500 { lock.logs.pop_front(); }
    lock.logs.push_back(msg);
    Ok(format!("USB -> {} @ {}", port, baud))
}

#[tauri::command]
fn set_wifi_interface(state: State<AppState>, iface: String) -> Result<String, String> {
    let mut lock = state.engine.state.lock();
    lock.device_config.wifi_interface = iface.clone();
    let ts = chrono::Local::now().format("%H:%M:%S").to_string();
    let msg = format!("[{}] DEVICE: WiFi interface set to {}", ts, iface);
    if lock.logs.len() >= 500 { lock.logs.pop_front(); }
    lock.logs.push_back(msg);
    Ok(format!("WiFi interface -> {}", iface))
}

// --- Live Mode (main.py: toggle_live_mode) ---

#[tauri::command]
fn set_live_mode(state: State<AppState>, active: bool) -> Result<String, String> {
    let mut lock = state.engine.state.lock();
    lock.live_mode = active;
    let status = if active { "ON" } else { "OFF" };
    let ts = chrono::Local::now().format("%H:%M:%S").to_string();
    let msg = format!("[{}] LIVE MODE -> {}", ts, status);
    if lock.logs.len() >= 500 { lock.logs.pop_front(); }
    lock.logs.push_back(msg);
    Ok(format!("Live mode -> {}", status))
}

// --- Metrics (main.py: update_gui -> engine.get_metrics()) ---

#[tauri::command]
fn get_metrics(state: State<AppState>) -> Result<String, String> {
    state.engine.get_metrics_internal()
        .map_err(|e| format!("Metrics error: {}", e))
}

// --- Config Save/Load (config.py: save_config, load_config) ---

#[tauri::command]
fn save_config(state: State<AppState>) -> Result<String, String> {
    let lock = state.engine.state.lock();

    let mut guitar_config = HashMap::new();
    for (name, gs) in &lock.guitar_states {
        guitar_config.insert(name.clone(), GuitarConfigEntry { enabled: gs.enabled });
    }

    let config = NullMagnetConfig {
        version: "1.0.0".into(),
        branding: "NullMagnet Live".into(),
        audio_device_index: lock.device_config.audio_device_index,
        audio_gain: lock.device_config.audio_gain,
        audio_sample_rate: 44100,
        camera_device_indices: vec![lock.device_config.camera_device_index],
        camera_resolution: (320, 240),
        usb_serial_ports: if lock.device_config.usb_serial_port.is_empty() {
            vec![]
        } else {
            vec![lock.device_config.usb_serial_port.clone()]
        },
        usb_serial_baud: lock.device_config.usb_serial_baud,
        wifi_interface: lock.device_config.wifi_interface.clone(),
        guitar_config,
        headscale_enabled: lock.headscale.enabled,
        headscale_target_ip: lock.headscale.target_ip.clone(),
        headscale_target_port: lock.headscale.target_port,
        live_mode_default: lock.live_mode,
        uplink_ip: lock.uplink_url.replace("http://", "").split(':').next()
            .unwrap_or("192.168.1.19").into(),
        uplink_port: 8000,
        p2p_port: lock.p2p_config.listen_port,
        mint_threshold_bits: 256,
        gpu_cuda_device_id: 0,
        gpu_ocl_platform_id: 0,
        gpu_ocl_device_id: 0,
    };

    drop(lock);

    match serde_json::to_string_pretty(&config) {
        Ok(json) => {
            match fs::write(CONFIG_FILE, &json) {
                Ok(_) => Ok(format!("Saved: {}", CONFIG_FILE)),
                Err(e) => Err(format!("Write error: {}", e)),
            }
        }
        Err(e) => Err(format!("Serialize error: {}", e)),
    }
}

#[tauri::command]
fn load_config(state: State<AppState>) -> Result<String, String> {
    let json = fs::read_to_string(CONFIG_FILE)
        .map_err(|e| format!("Read error: {}", e))?;
    let config: NullMagnetConfig = serde_json::from_str(&json)
        .map_err(|e| format!("Parse error: {}", e))?;

    let mut lock = state.engine.state.lock();
    lock.device_config.audio_device_index = config.audio_device_index;
    lock.device_config.audio_gain = config.audio_gain;
    lock.device_config.camera_device_index = *config.camera_device_indices.first().unwrap_or(&0);
    lock.device_config.usb_serial_port = config.usb_serial_ports.first()
        .cloned().unwrap_or_default();
    lock.device_config.usb_serial_baud = config.usb_serial_baud;
    lock.device_config.wifi_interface = config.wifi_interface;
    lock.headscale.enabled = config.headscale_enabled;
    lock.headscale.target_ip = config.headscale_target_ip;
    lock.headscale.target_port = config.headscale_target_port;
    lock.live_mode = config.live_mode_default;
    lock.uplink_url = format!("http://{}:{}/entropy", config.uplink_ip, config.uplink_port);
    lock.p2p_config.listen_port = config.p2p_port;

    // Restore guitar enable states
    for (name, entry) in &config.guitar_config {
        if let Some(gs) = lock.guitar_states.get_mut(name) {
            gs.enabled = entry.enabled;
        }
    }

    Ok(format!("Loaded: {}", CONFIG_FILE))
}

// --- Shutdown (main.py: engine.shutdown()) ---

#[tauri::command]
fn shutdown(state: State<AppState>) -> Result<String, String> {
    state.engine.shutdown_internal();
    Ok("Shutdown complete".into())
}

// ============================================================================
// TAURI APP ENTRY POINT
// ============================================================================

fn main() {
    // Create keys directory
    let _ = fs::create_dir_all("keys");

    // Initialize the engine (native Rust, no PyO3)
    let engine = ChaosEngine::new_native();

    // Auto-load config if it exists
    if std::path::Path::new(CONFIG_FILE).exists() {
        if let Ok(json) = fs::read_to_string(CONFIG_FILE) {
            if let Ok(config) = serde_json::from_str::<NullMagnetConfig>(&json) {
                let mut lock = engine.state.lock();
                lock.device_config.audio_device_index = config.audio_device_index;
                lock.device_config.audio_gain = config.audio_gain;
                lock.headscale.enabled = config.headscale_enabled;
                lock.headscale.target_ip = config.headscale_target_ip;
                lock.headscale.target_port = config.headscale_target_port;
                lock.live_mode = config.live_mode_default;
                lock.p2p_config.listen_port = config.p2p_port;
                println!("INFO: Loaded config from {}", CONFIG_FILE);
            }
        }
    }

    // Auto-load HMAC key from environment if available
    if let Ok(key_hex) = std::env::var("NULL_P2P_HMAC_KEY_HEX") {
        if !key_hex.is_empty() {
            if let Ok(key_bytes) = hex::decode(&key_hex) {
                if key_bytes.len() == 32 {
                    engine.state.lock().p2p_config.hmac_key = Some(key_bytes);
                    println!("INFO: Loaded P2P HMAC key from environment");
                }
            }
        }
    }

    println!("NullMagnet Live v1.0 - Jupiter Labs");
    println!("NIST SP 800-90B Aligned Entropy Harvesting");
    println!("Tauri GUI starting...");

    tauri::Builder::default()
        .manage(AppState { engine })
        .invoke_handler(tauri::generate_handler![
            // Harvester controls
            toggle_harvester,
            toggle_guitar,
            // Network
            toggle_uplink,
            set_network_target,
            set_uplink_target,
            toggle_headscale,
            set_headscale_target,
            // P2P
            toggle_p2p,
            set_p2p_port,
            add_peer,
            set_p2p_hmac_key,
            load_hmac_from_env,
            // PQC
            mint_pqc_bundle,
            set_auto_mint,
            // Device manager
            list_audio_devices,
            list_camera_devices,
            list_serial_ports,
            set_audio_device,
            set_audio_gain,
            set_camera_device,
            set_usb_serial_port,
            set_wifi_interface,
            // Live mode
            set_live_mode,
            // Metrics (called every 100ms by frontend)
            get_metrics,
            // Config
            save_config,
            load_config,
            // Shutdown
            shutdown,
        ])
        .run(tauri::generate_context!())
        .expect("Failed to run NullMagnet Live");
}
