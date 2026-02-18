# NULLMAGNET LIVE v1.0

## Jupiter Labs -- NIST SP 800-90B Aligned Entropy Harvesting Platform

------------------------------------------------------------------------

## Overview

NullMagnet Live is a real-time entropy harvesting, conditioning,
validation, and post-quantum key minting platform built under Jupiter
Labs. It integrates multiple heterogeneous entropy sources including
system noise, GPU jitter, USB serial streams, WiFi environmental
variation, audio, video, HID activity, and distributed ESP32-based
guitar nodes.

The system is aligned to NIST SP 800-90B health testing principles and
is designed for high-assurance entropy collection, conditioning, and
PQC-based key generation.

This README documents architecture, configuration, runtime behavior, GUI
structure, network topology, and operational guidance.

------------------------------------------------------------------------

# Table of Contents

1.  Architecture Overview\
2.  Configuration Model\
3.  Entropy Sources\
4.  Guitar ESP32 Integration\
5.  NIST Health Testing\
6.  Conditioning & Extraction\
7.  Post-Quantum Cryptography\
8.  Network & Headscale Integration\
9.  P2P Entropy Sharing\
10. GUI Architecture\
11. Device Management\
12. Build & Deployment\
13. Security Model\
14. Operational Workflow\
15. Future Expansion\
16. File Reference Summary

------------------------------------------------------------------------

# 1. Architecture Overview

NullMagnet Live consists of:

-   Rust core entropy engine
-   Tauri desktop wrapper
-   Web-based UI (HTML/CSS/JS)
-   Config-driven runtime behavior
-   Optional distributed entropy nodes (ESP32 guitars)
-   Optional Headscale tailnet forwarding
-   PQC bundle minting (Kyber/Falcon aligned)

Data flow:

Raw Sources → Health Tests → Conditioning → Extraction Pool → Credited
Bits → PQC Minting

------------------------------------------------------------------------

# 2. Configuration Model

Primary runtime configuration is defined in:

-   `nullmagnet_config.json`

Key configuration domains:

-   Headscale routing
-   Guitar UDP ports
-   Local sensor toggles
-   GPU priority
-   Network uplink
-   NIST health parameters
-   Auto-mint thresholds

Example categories include:

## Headscale

-   enabled
-   Aoi Midori IP
-   UDP forwarding

## Guitar Sources

Each guitar defines:

-   Label
-   UDP ping port
-   Entropy port
-   Enabled state

## Local Sensors

-   Audio auto-detection
-   Video max simultaneous feeds
-   GPU backend priority order
-   USB serial auto-detection
-   WiFi polling interval

## NIST Parameters

-   Repetition Count Test cutoff
-   Adaptive Proportion Test window and cutoff
-   Conditioning factor
-   Startup discard
-   Minimum entropy threshold for minting

------------------------------------------------------------------------

# 3. Entropy Sources

Supported entropy domains:

-   System / CPU jitter
-   Hardware TRNG
-   Audio microphone entropy
-   Video frame entropy
-   Mouse HID timing
-   CUDA GPU noise
-   OpenCL GPU noise
-   WiFi environmental variance
-   USB serial byte stream entropy
-   Guitar ESP32 UDP entropy

Each source tracks:

-   Shannon entropy estimate
-   Min-entropy estimate
-   Health state
-   Sample count

------------------------------------------------------------------------

# 4. Guitar ESP32 Integration

Each guitar node (Spectra, Neptonius, Thalyn) provides:

-   UDP ping channel
-   UDP entropy channel
-   Packet tracking
-   Real-time status display

Ports are defined in configuration and mapped directly into GUI display
and backend listener.

------------------------------------------------------------------------

# 5. NIST Health Testing

Implemented health tests:

-   Repetition Count Test (RCT)
-   Adaptive Proportion Test (APT)

Parameters are configurable.

Health states include:

-   INIT
-   STARTUP
-   STEADY
-   FAILED
-   DEAD

Startup discard is applied before entropy is credited.

------------------------------------------------------------------------

# 6. Conditioning & Extraction

Entropy conditioning uses SHA-256-based mixing.

Extraction pool tracks:

-   Accumulated raw bytes
-   Extracted bytes
-   Compression ratio
-   Credited bits

Pool state is displayed as conditioned SHA-256 hex digest.

------------------------------------------------------------------------

# 7. Post-Quantum Cryptography

Minting threshold defaults to 256 credited bits.

PQC readiness enables:

-   Kyber (ML-KEM)
-   Falcon signatures

Minting produces PQC bundles upon threshold completion or manual
request.

------------------------------------------------------------------------

# 8. Network & Headscale Integration

Uplink supports forwarding entropy metrics to a designated IP.

Headscale forwarding allows integration with tailnet systems such as Aoi
Midori node.

Network states:

-   INIT
-   ONLINE
-   OFFLINE

------------------------------------------------------------------------

# 9. P2P Entropy Sharing

Optional P2P module:

-   Configurable listen port
-   Peer registration
-   Optional HMAC authentication
-   Peer count tracking
-   Received packet count

------------------------------------------------------------------------

# 10. GUI Architecture

The GUI is built using:

-   HTML
-   CSS with structured variable theming
-   JavaScript metrics polling
-   Tauri invoke bindings

Tabs include:

-   Sources
-   Network
-   Devices

Panels include:

-   Real-time entropy graph
-   Pool state display
-   Audit log
-   Vault control
-   Device enumeration

------------------------------------------------------------------------

# 11. Device Management

Device enumeration includes:

-   Audio input list
-   Camera list
-   USB serial port list
-   WiFi interface selection

Auto-scan available on tab entry.

------------------------------------------------------------------------

# 12. Build & Deployment

Project uses:

-   Tauri CLI
-   Rust backend
-   Node package definitions

Scripts:

-   tauri
-   tauri dev
-   tauri build

Targets include Windows bundling with defined identifier.

------------------------------------------------------------------------

# 13. Security Model

Security principles:

-   No single entropy source trusted
-   Multi-source blending
-   Health test enforcement
-   SHA-256 conditioning
-   Optional HMAC for P2P
-   Post-quantum key generation
-   Config scope restrictions in Tauri allowlist

------------------------------------------------------------------------

# 14. Operational Workflow

Typical flow:

1.  Enable sources
2.  Observe entropy graph
3.  Validate health states
4.  Accumulate credited bits
5.  Mint PQC bundle
6.  Optionally forward to network

Live Mode provides pulsed visual feedback on extraction events.

------------------------------------------------------------------------

# 15. Future Expansion

Planned expansion domains:

-   Additional hardware entropy nodes
-   Cloud chamber integration
-   Satellite anomaly input
-   Advanced PQC variants
-   Distributed entropy mesh
-   Audit export systems

------------------------------------------------------------------------

# 16. File Reference Summary

-   nullmagnet_config.json\
-   package.json\
-   tauri.conf.json\
-   index.html

------------------------------------------------------------------------

Generated: 2026-02-18T03:43:36.400864 UTC
