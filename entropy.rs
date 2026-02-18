//! NullMagnet Live - entropy.rs
//! NIST SP 800-90B Health Testing & Entropy Calculations
//!
//! Jupiter Labs - Modular entropy engine
//!
//! This module contains:
//! - NIST health test state machine (RCT + APT)
//! - Shannon entropy, min-entropy, collision entropy estimators
//! - Extraction pool with SHA-256 conditioning
//! - Conservative entropy crediting (0.85 factor per NIST)
//!
//! References:
//! - NIST SP 800-90B Section 4.4.1: Repetition Count Test
//! - NIST SP 800-90B Section 4.4.2: Adaptive Proportion Test
//! - NIST SP 800-90B Section 3.1.5: Conditioning factor
//! - NIST SP 800-90B Section 6: Min-entropy estimation

use std::collections::VecDeque;
use sha2::{Sha256, Digest as Sha2Digest};
use std::time::{SystemTime, UNIX_EPOCH};

// ============================================================================
// NIST SP 800-90B CONSTANTS
// ============================================================================

/// Extraction pool size (raw bytes before conditioning)
pub const EXTRACTION_POOL_SIZE: usize = 256;

/// Internal pool display size
pub const POOL_SIZE: usize = 1024;

/// History length for graphs
pub const HISTORY_LEN: usize = 300;

/// NIST RCT: False positive rate alpha = 2^-30
/// For H=1 bit/byte: C = 1 + ceil(30/1) = 31
/// Verified against EIP130 TRNG SP 800-90B public use document
pub const NIST_RCT_CUTOFF: u32 = 31;

/// NIST APT: Window size for non-binary data
/// Minimum recommended W=512 for non-binary sources (Section 4.4.2)
pub const NIST_APT_WINDOW: usize = 512;

/// NIST APT: Cutoff for alpha = 2^-30, H=1, W=512
/// C = 325 (verified against NIST reference implementations)
pub const NIST_APT_CUTOFF: u32 = 325;

/// Startup test: samples to discard after power-on
/// NIST requires startup health tests; 4096 provides safe margin
pub const STARTUP_DISCARD_SAMPLES: usize = 4096;

/// NIST conditioning factor (0.85 per SP 800-90B Section 3.1.5)
/// Conditioned output gets credited at 85% of input entropy
pub const NIST_CONDITIONING_FACTOR: f64 = 0.85;

/// Minimum credited entropy bits before key minting
pub const MIN_ENTROPY_FOR_MINT: f64 = 256.0;

/// Auto-mint quality threshold (min-entropy per byte)
/// Lowered from 6.0 to 1.0 - most real sources produce 1-4 bits/byte;
/// 6.0 was too strict and auto-mint would never trigger.
pub const AUTO_MINT_THRESHOLD: f64 = 1.0;

// ============================================================================
// NIST HEALTH TEST STATE MACHINE
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HealthState {
    Init,
    Startup,    // Running startup tests, samples discarded
    Steady,     // Normal operation
    Failed,     // Source failed health test, awaiting recovery
    Dead,       // Max retries exceeded, permanently disabled
}

/// Maximum health test failures before source is permanently disabled
pub const MAX_HEALTH_RETRIES: u32 = 5;

/// Samples to wait before attempting health test recovery
pub const RECOVERY_COOLDOWN_SAMPLES: u64 = 10000;

#[derive(Clone)]
pub struct NistHealthTester {
    pub state: HealthState,

    // RCT state
    rct_last_value: Option<u8>,
    rct_count: u32,

    // APT state
    apt_window: VecDeque<u8>,
    apt_first_value: Option<u8>,
    apt_count: u32,

    // Startup tracking
    startup_samples: usize,

    // Recovery tracking
    failure_count: u32,
    samples_since_failure: u64,

    // Statistics
    pub total_samples: u64,
    pub rct_failures: u64,
    pub apt_failures: u64,
    pub samples_passed: u64,
}

impl NistHealthTester {
    pub fn new() -> Self {
        Self {
            state: HealthState::Init,
            rct_last_value: None,
            rct_count: 0,
            apt_window: VecDeque::with_capacity(NIST_APT_WINDOW),
            apt_first_value: None,
            apt_count: 0,
            startup_samples: 0,
            failure_count: 0,
            samples_since_failure: 0,
            total_samples: 0,
            rct_failures: 0,
            apt_failures: 0,
            samples_passed: 0,
        }
    }

    pub fn start(&mut self) {
        self.state = HealthState::Startup;
        self.reset();
    }

    pub fn reset(&mut self) {
        self.rct_last_value = None;
        self.rct_count = 0;
        self.apt_window.clear();
        self.apt_first_value = None;
        self.apt_count = 0;
        self.startup_samples = 0;
    }

    pub fn trigger_on_demand(&mut self) {
        self.state = HealthState::Startup;
        self.reset();
    }

    /// Process a batch of samples through health tests.
    /// Returns samples that passed (empty during startup, failed, or dead).
    pub fn process_batch(&mut self, data: &[u8]) -> Vec<u8> {
        // Dead sources never recover - require manual intervention
        if self.state == HealthState::Dead {
            return Vec::new();
        }

        // Failed sources attempt automatic recovery after cooldown
        if self.state == HealthState::Failed {
            self.samples_since_failure += data.len() as u64;
            if self.samples_since_failure >= RECOVERY_COOLDOWN_SAMPLES {
                if self.failure_count >= MAX_HEALTH_RETRIES {
                    self.state = HealthState::Dead;
                    return Vec::new();
                }
                // Attempt recovery: reset to startup tests
                self.state = HealthState::Startup;
                self.reset();
                self.samples_since_failure = 0;
            } else {
                return Vec::new();
            }
        }

        let mut passed = Vec::with_capacity(data.len());

        for &sample in data {
            self.total_samples += 1;

            // Run Repetition Count Test (NIST SP 800-90B Section 4.4.1)
            if !self.run_rct(sample) {
                self.state = HealthState::Failed;
                self.rct_failures += 1;
                self.failure_count += 1;
                self.samples_since_failure = 0;
                return Vec::new();
            }

            // Run Adaptive Proportion Test (NIST SP 800-90B Section 4.4.2)
            if !self.run_apt(sample) {
                self.state = HealthState::Failed;
                self.apt_failures += 1;
                self.failure_count += 1;
                self.samples_since_failure = 0;
                return Vec::new();
            }

            // Handle state transitions
            match self.state {
                HealthState::Init => {
                    // Not started yet - should not normally reach here
                }
                HealthState::Startup => {
                    self.startup_samples += 1;
                    if self.startup_samples >= STARTUP_DISCARD_SAMPLES {
                        self.state = HealthState::Steady;
                        // Reset failure count on successful startup (source recovered)
                        if self.failure_count > 0 {
                            self.failure_count = 0;
                        }
                    }
                    // Discard startup samples per NIST requirement
                }
                HealthState::Steady => {
                    passed.push(sample);
                    self.samples_passed += 1;
                }
                HealthState::Failed | HealthState::Dead => {
                    return Vec::new();
                }
            }
        }

        passed
    }

    /// NIST SP 800-90B Section 4.4.1 - Repetition Count Test
    ///
    /// Detects when the source gets stuck on one output value.
    /// If the same value repeats C (=31) times consecutively, the test fails.
    fn run_rct(&mut self, sample: u8) -> bool {
        match self.rct_last_value {
            Some(last) if last == sample => {
                self.rct_count += 1;
                if self.rct_count >= NIST_RCT_CUTOFF {
                    return false;
                }
            }
            _ => {
                self.rct_last_value = Some(sample);
                self.rct_count = 1;
            }
        }
        true
    }

    /// NIST SP 800-90B Section 4.4.2 - Adaptive Proportion Test
    ///
    /// Detects when one value becomes much more common than expected.
    /// Uses a sliding window of W (=512) samples; if the first sample's
    /// count in the window reaches C (=325), the test fails.
    fn run_apt(&mut self, sample: u8) -> bool {
        self.apt_window.push_back(sample);

        if self.apt_first_value.is_none() {
            self.apt_first_value = Some(sample);
            self.apt_count = 1;
            return true;
        }

        if Some(sample) == self.apt_first_value {
            self.apt_count += 1;
            if self.apt_count >= NIST_APT_CUTOFF {
                return false;
            }
        }

        // Reset window when full
        if self.apt_window.len() >= NIST_APT_WINDOW {
            self.apt_window.clear();
            self.apt_first_value = None;
            self.apt_count = 0;
        }

        true
    }

    pub fn is_healthy(&self) -> bool {
        self.state == HealthState::Steady
    }

    pub fn is_dead(&self) -> bool {
        self.state == HealthState::Dead
    }

    pub fn failure_count(&self) -> u32 {
        self.failure_count
    }

    pub fn state_name(&self) -> &'static str {
        match self.state {
            HealthState::Init => "INIT",
            HealthState::Startup => "STARTUP",
            HealthState::Steady => "STEADY",
            HealthState::Failed => "FAILED",
            HealthState::Dead => "DEAD",
        }
    }
}

// ============================================================================
// ENTROPY CALCULATIONS (NIST SP 800-90B Section 6)
// ============================================================================

pub fn get_timestamp() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs()
}

pub fn get_timestamp_nanos() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos() as u64
}

/// Shannon entropy (upper bound, NOT used for crediting - informational only)
pub fn shannon_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut entropy = 0.0;
    let mut counts = [0usize; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let len = data.len() as f64;
    for &count in &counts {
        if count > 0 {
            let p = count as f64 / len;
            entropy -= p * p.log2();
        }
    }
    entropy
}

/// NIST Min-entropy (Most Common Value estimate)
/// H_min = -log2(p_max)
/// This is the primary entropy measure per NIST SP 800-90B
pub fn min_entropy(data: &[u8]) -> f64 {
    if data.is_empty() {
        return 0.0;
    }
    let mut counts = [0usize; 256];
    for &b in data {
        counts[b as usize] += 1;
    }
    let max_count = counts.iter().max().copied().unwrap_or(0);
    let max_prob = max_count as f64 / data.len() as f64;
    if max_prob <= 0.0 || max_prob >= 1.0 {
        return 0.0;
    }
    -max_prob.log2()
}

/// NIST Collision estimate (better for non-IID sources)
/// Estimates entropy based on expected distance to first collision
pub fn collision_entropy(data: &[u8]) -> f64 {
    if data.len() < 10 {
        return 0.0;
    }

    let mut collision_sum = 0u64;
    let mut collision_count = 0u64;

    for i in 0..data.len().saturating_sub(2).min(1000) {
        let target = data[i];
        for j in (i + 1)..data.len().min(i + 100) {
            if data[j] == target {
                collision_sum += (j - i) as u64;
                collision_count += 1;
                break;
            }
        }
    }

    if collision_count == 0 {
        return 8.0;
    }

    let mean_collision = collision_sum as f64 / collision_count as f64;
    let p_estimate = 1.0 / mean_collision;
    if p_estimate <= 0.0 || p_estimate >= 1.0 {
        return 0.0;
    }

    -p_estimate.log2()
}

/// Conservative combined min-entropy estimate
/// Takes the minimum of MCV and collision estimates (most conservative)
pub fn conservative_min_entropy(data: &[u8]) -> f64 {
    let mcv = min_entropy(data);
    let coll = collision_entropy(data);
    mcv.min(coll).max(0.0).min(8.0)
}

/// Credit entropy with NIST 0.85 conditioning factor
/// Per NIST SP 800-90B Section 3.1.5.1.1 (vetted conditioning):
///   h_out = min(h_in, n_out, q) * 0.85
/// where:
///   h_in  = total input entropy bits (raw_bytes * min_ent_per_byte)
///   n_out = conditioned output size in bits (256 for SHA-256)
///   q     = narrowest internal width in bits (256 for SHA-256)
///
/// CRITICAL: Output entropy is CAPPED at min(n_out, q) = 256 bits
/// regardless of how much input entropy is accumulated.
pub fn credit_entropy(raw_bytes: usize, min_ent_per_byte: f64) -> f64 {
    let h_in = raw_bytes as f64 * min_ent_per_byte;
    // SHA-256: n_out = 256 bits, q (narrowest internal width) = 256 bits
    let n_out: f64 = 256.0;
    let q: f64 = 256.0;
    let capped = h_in.min(n_out).min(q);
    capped * NIST_CONDITIONING_FACTOR
}

// ============================================================================
// ENTROPY EXTRACTION POOL (SHA-256 Conditioning)
// ============================================================================

#[derive(Clone)]
pub struct EntropyExtractionPool {
    pub buffer: Vec<u8>,
    pub extractions_count: u64,
    pub last_extraction: f64,
    pub total_raw_consumed: usize,
    pub total_extracted_bytes: usize,
    pub credited_entropy_bits: f64,  // NIST-credited bits accumulated

    // Conditioned aggregate entropy tracking
    // Tracks credited bits across ALL sources combined (not per-extraction)
    pub aggregate_credited_bits: f64,
}

impl EntropyExtractionPool {
    pub fn new() -> Self {
        Self {
            buffer: Vec::with_capacity(EXTRACTION_POOL_SIZE),
            extractions_count: 0,
            last_extraction: 0.0,
            total_raw_consumed: 0,
            total_extracted_bytes: 0,
            credited_entropy_bits: 0.0,
            aggregate_credited_bits: 0.0,
        }
    }

    /// Add raw bytes to the extraction pool with entropy tracking.
    /// Returns all extracted outputs when the pool fills (may extract multiple times).
    pub fn add_raw_bytes(&mut self, raw_data: &[u8], min_ent: f64) -> Option<Vec<u8>> {
        self.buffer.extend_from_slice(raw_data);

        // Track credited entropy for this accumulation cycle
        let credited = credit_entropy(raw_data.len(), min_ent);
        self.credited_entropy_bits += credited;
        self.aggregate_credited_bits += credited;

        // Extract as many times as needed (handles data > EXTRACTION_POOL_SIZE)
        let mut all_extracted = Vec::new();
        while self.buffer.len() >= EXTRACTION_POOL_SIZE {
            let extracted = self.extract();
            all_extracted.extend_from_slice(&extracted);
        }

        if all_extracted.is_empty() {
            None
        } else {
            Some(all_extracted)
        }
    }

    /// Extract conditioned output using SHA-256 (vetted conditioning function)
    /// Per NIST SP 800-90B, SHA-256 is an approved unkeyed conditioning component
    /// Only consumes EXTRACTION_POOL_SIZE bytes; remainder stays in buffer.
    fn extract(&mut self) -> Vec<u8> {
        // Take exactly EXTRACTION_POOL_SIZE bytes for conditioning
        let consume_len = EXTRACTION_POOL_SIZE.min(self.buffer.len());
        let to_condition: Vec<u8> = self.buffer.drain(..consume_len).collect();

        let mut hasher = Sha256::new();
        hasher.update(&to_condition);
        hasher.update(&self.extractions_count.to_le_bytes());
        hasher.update(&get_timestamp_nanos().to_le_bytes());
        let result = hasher.finalize();

        self.total_raw_consumed += consume_len;
        self.total_extracted_bytes += 32;

        self.extractions_count += 1;
        self.last_extraction = get_timestamp() as f64;

        // Reset per-cycle credited bits after extraction
        // (aggregate_credited_bits is NOT reset here - it accumulates globally)
        self.credited_entropy_bits = 0.0;

        result.to_vec()
    }

    pub fn fill_percentage(&self) -> f64 {
        (self.buffer.len() as f64 / EXTRACTION_POOL_SIZE as f64) * 100.0
    }

    pub fn accumulated_bytes(&self) -> usize {
        self.buffer.len()
    }

    /// Reset aggregate credited bits (called after key minting)
    pub fn reset_aggregate_credits(&mut self) {
        self.aggregate_credited_bits = 0.0;
    }
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rct_pass_normal_data() {
        let mut ht = NistHealthTester::new();
        ht.start();
        // Random-ish data should pass RCT
        let data: Vec<u8> = (0..=255).cycle().take(5000).collect();
        let passed = ht.process_batch(&data);
        // First 4096 discarded (startup), rest should pass
        assert!(passed.len() > 0);
        assert_eq!(ht.state, HealthState::Steady);
    }

    #[test]
    fn test_rct_fail_stuck_source() {
        let mut ht = NistHealthTester::new();
        ht.start();
        // 31+ identical values should fail RCT
        let data = vec![42u8; 50];
        let passed = ht.process_batch(&data);
        assert!(passed.is_empty());
        assert_eq!(ht.state, HealthState::Failed);
        assert_eq!(ht.failure_count(), 1);
    }

    #[test]
    fn test_health_recovery_after_failure() {
        let mut ht = NistHealthTester::new();
        ht.start();
        // Trigger a failure
        let stuck = vec![42u8; 50];
        let _ = ht.process_batch(&stuck);
        assert_eq!(ht.state, HealthState::Failed);

        // Feed enough samples for cooldown (but they're discarded)
        let cooldown: Vec<u8> = (0..=255).cycle().take(RECOVERY_COOLDOWN_SAMPLES as usize).collect();
        let passed = ht.process_batch(&cooldown);
        // Should have transitioned to Startup, then eventually Steady
        assert!(ht.state == HealthState::Startup || ht.state == HealthState::Steady);
    }

    #[test]
    fn test_health_dead_after_max_retries() {
        let mut ht = NistHealthTester::new();
        ht.start();

        for _ in 0..MAX_HEALTH_RETRIES {
            // Trigger failure
            let stuck = vec![42u8; 50];
            let _ = ht.process_batch(&stuck);
            assert_eq!(ht.state, HealthState::Failed);

            // Feed cooldown to allow recovery attempt
            let cooldown: Vec<u8> = (0..=255).cycle().take(RECOVERY_COOLDOWN_SAMPLES as usize).collect();
            let _ = ht.process_batch(&cooldown);
        }

        // One more failure should push to Dead
        let stuck = vec![42u8; 50];
        let _ = ht.process_batch(&stuck);

        // Feed cooldown - should transition to Dead, not Startup
        let cooldown: Vec<u8> = (0..=255).cycle().take(RECOVERY_COOLDOWN_SAMPLES as usize).collect();
        let passed = ht.process_batch(&cooldown);
        assert!(passed.is_empty());
        assert_eq!(ht.state, HealthState::Dead);
        assert!(ht.is_dead());
    }

    #[test]
    fn test_min_entropy_uniform() {
        let data: Vec<u8> = (0..=255).collect();
        let h = min_entropy(&data);
        // Uniform distribution: H_min = log2(256) = 8.0
        assert!((h - 8.0).abs() < 0.01);
    }

    #[test]
    fn test_min_entropy_biased() {
        // Heavily biased: 90% zeros
        let mut data = vec![0u8; 900];
        data.extend(vec![1u8; 100]);
        let h = min_entropy(&data);
        // H_min = -log2(0.9) ~= 0.152
        assert!(h < 1.0);
        assert!(h > 0.0);
    }

    #[test]
    fn test_credit_entropy() {
        // 100 bytes at 4.0 bits/byte = 400 bits input
        // But capped at min(400, 256, 256) = 256 bits
        // Credited = 256 * 0.85 = 217.6 bits
        let credited = credit_entropy(100, 4.0);
        assert!((credited - 217.6).abs() < 0.01);

        // Small input: 10 bytes at 2.0 bits/byte = 20 bits input
        // Capped at min(20, 256, 256) = 20 bits
        // Credited = 20 * 0.85 = 17.0 bits
        let credited_small = credit_entropy(10, 2.0);
        assert!((credited_small - 17.0).abs() < 0.01);

        // Maximum possible per extraction: 256 * 0.85 = 217.6
        let credited_max = credit_entropy(1000, 8.0);
        assert!((credited_max - 217.6).abs() < 0.01);
    }

    #[test]
    fn test_extraction_pool_cycle() {
        let mut pool = EntropyExtractionPool::new();
        // Fill pool to trigger extraction
        let data = vec![0xAB; EXTRACTION_POOL_SIZE];
        let result = pool.add_raw_bytes(&data, 4.0);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 32); // SHA-256 output
        assert_eq!(pool.extractions_count, 1);
        assert_eq!(pool.buffer.len(), 0); // Buffer should be empty
    }

    #[test]
    fn test_extraction_pool_overflow() {
        let mut pool = EntropyExtractionPool::new();
        // Send 3x the pool size - should produce 3 extractions (96 bytes)
        let data = vec![0xCD; EXTRACTION_POOL_SIZE * 3];
        let result = pool.add_raw_bytes(&data, 2.0);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 96); // 3 x 32 = 96
        assert_eq!(pool.extractions_count, 3);
        assert_eq!(pool.buffer.len(), 0); // No remainder
    }

    #[test]
    fn test_extraction_pool_remainder() {
        let mut pool = EntropyExtractionPool::new();
        // Send pool size + 50 extra bytes
        let data = vec![0xEF; EXTRACTION_POOL_SIZE + 50];
        let result = pool.add_raw_bytes(&data, 3.0);
        assert!(result.is_some());
        assert_eq!(result.unwrap().len(), 32); // 1 extraction
        assert_eq!(pool.extractions_count, 1);
        assert_eq!(pool.buffer.len(), 50); // 50 bytes remain
    }

    #[test]
    fn test_conditioning_factor_value() {
        // Verify NIST conditioning factor is exactly 0.85
        assert!((NIST_CONDITIONING_FACTOR - 0.85).abs() < f64::EPSILON);
    }

    #[test]
    fn test_nist_constants() {
        // Verify NIST constants match published values
        assert_eq!(NIST_RCT_CUTOFF, 31);
        assert_eq!(NIST_APT_WINDOW, 512);
        assert_eq!(NIST_APT_CUTOFF, 325);
        assert_eq!(STARTUP_DISCARD_SAMPLES, 4096);
    }
}
