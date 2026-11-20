# NALA-airllm-rs (`v4.0.0-nala.6.6.6`)

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-10b981.svg)](./LICENSE)
[![Rust Edition 2024](https://img.shields.io/badge/Rust-Edition_2024_%7C_Zero_Unsafe-06b6d4.svg)](./Cargo.toml)
[![Standalone Pure Rust](https://img.shields.io/badge/Standalone-Zero_Python_%7C_Zero_Node.js_%7C_0_Crates-f59e0b.svg)](./src/main.rs)
[![GitHub Release](https://img.shields.io/github/v/release/Master-MD/NALA-airllm-rs?color=10b981&label=Release)](https://github.com/Master-MD/NALA-airllm-rs/releases/latest)
[![Multi-OS CI](https://github.com/Master-MD/NALA-airllm-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/Master-MD/NALA-airllm-rs/actions/workflows/ci.yml)

**Sprachen / Languages / Linguas:**  
[**Deutsch (`DE`)**](./docs/README_DE.md) · [**English (`EN`)**](./docs/README_EN.md) · [**Français (`FR`)**](./docs/README_FR.md) · [**Italiano (`IT`)**](./docs/README_IT.md) · [**Rumantsch (`RM`)**](./docs/README_RM.md) · [**5-Language License Summary**](./LICENSE_SUMMARY_5LANG.md) · [**60s Caveman Guide**](./CAVEMAN_GUIDE.md) · [**FAQ**](./FAQ.md) · [**Security**](./SECURITY.md) · [**Contributing**](./CONTRIBUTING.md)

---

**`NALA-airllm-rs`** (`nala-airllm-rs` / alias `airllm-rs`) is a **100% Native Pure Safe Rust (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` external crates, Zero-Python, Zero-Node.js, Zero-Telemetry)** clean-room rewrite of **[`lyogavin/airllm`](https://github.com/lyogavin/airllm)**.

Run **70B to 2.8T parameter models on a single 4 GB GPU** and fine-tune **27B to 125B models via Streamed LoRA under 6 GB VRAM** — packaged in a single ~750 KB native binary with zero Python/PyTorch/CUDA wheel hell.

- **Apache-2.0 Derivative Work:** Full attribution preserved in [`LICENSE`](./LICENSE), [`NOTICE`](./NOTICE), [`THIRD_PARTY_NOTICES.md`](./THIRD_PARTY_NOTICES.md), and [`README_UPSTREAM_AIRLLM.md`](./README_UPSTREAM_AIRLLM.md).
- **Cross-OS Parity (macOS Restriction Lifted):** Runs natively on **macOS (Apple Silicon & Intel)**, **Linux (`x86_64` & `arm64`)**, and **Windows (`x86_64`)** across **all** supported architectures (lifting upstream AirLLM's macOS Llama/Mistral-only restriction).

---

## Complete Feature Parity & Rust-Turbo Upgrades

| Capability | Upstream Python AirLLM | `NALA-airllm-rs v4.0.0-nala.6.6.6` (Pure Safe Rust) |
| :--- | :--- | :--- |
| **Runtime Dependencies** | Python + PyTorch + `transformers` + `accelerate` + `bitsandbytes` + `safetensors` + `mlx` | **0 dependencies** (`std` only, `#![forbid(unsafe_code)]`, ~752 KB binary) |
| **AutoModel & Architecture Overrides** | Generic `*ForCausalLM` + ChatGLM, QWen, Baichuan, InternLM, Kimi K3, Qwen3.5, Qwen4Exp | **100% parity** + nested multimodal `text_config`, `tie_word_embeddings`, `mtp.*` drop, `int64` scatter guard (`src/catalog.rs`) |
| **Safetensors Layer Splitter** | Single-file & multi-shard index, ascending shard order, hardlink passthrough | **100% native binary parser/writer** + longest-prefix `layer_owner()`, zero-copy hardlink passthrough, `--delete-original` (`src/safetensors.rs`) |
| **Per-Expert MoE Streaming** | Kimi K3 (896 experts / 16 routed, ~1 GB vs 55 GB) | **Direct byte-offset random-access seek** (`load_safetensors_subset`) + Top-K router (`src/streamer.rs`) |
| **PLE N-Gram Host Mmap** | Qwen3.8-Flash-Next 51B (~102 GB) host table | **Contiguous shard merger** -> flat `.mmap` + `.mmap.json` + direct row-seek `gather_ngram_rows()` (0 bytes in VRAM) |
| **Block-Wise Compression** | `4bit` NF4 & `8bit` via CUDA-only `bitsandbytes` | **Cross-OS Pure Safe Rust** NF4 (`blocksize=64`, **3.56x**), INT8 (`blocksize=2048`), OCP MXFP4 (`E2M1+E8M0`) GEMV & FP8 verbatim guard (`src/quant.rs`) |
| **Double-Buffered Prefetch** | `ThreadPoolExecutor` + 2 GiB pinned cap | **Lock-free bounded `sync_channel` worker thread** + `LayeredProfiler` microsecond telemetry (`src/streamer.rs`) |
| **Streamed LoRA + Chunked CE** | `AirLLMLoRA`, `AirLLMLoRAQwen4Exp`, `chunked_linear_cross_entropy`, SFT loader | **Pure Rust `LoRALinear` + `AdamW`**, exact online-`logaddexp` `chunked_linear_cross_entropy_fwd_bwd` (finite-diff verified), full `.jsonl`/`.json`/`.txt` SFT loader with `-100` masking (`src/lora.rs`) |
| **macOS Support** | Apple Silicon + Llama/Mistral only | **All architectures unlocked** on Apple Silicon & Intel (`src/hal.rs`) |
| **Security, UI, MCP & Evolution** | None | **sHeriFF `SEC-01..05` + Zero-PII Gate**, **12-Tool JSON-RPC 2.0 MCP Server**, **OpenAI-compatible `127.0.0.1` Server**, **Offline AAA 2-Click Dream/REA/RSI Cockpit** & **Multi-IDE Prompt Compiler** (`src/security.rs`, `src/mcp.rs`, `src/server.rs`, `src/dream_rsi.rs`) |

---

## Real Measured Hardware Benchmarks (`nala-airllm-rs bench`)

Measured directly with `RUSTFLAGS="-D warnings" cargo build --release` (`nala-airllm-rs v4.0.0-nala.6.6.6`, Apple Silicon `arm64`, 24/24 tests passing in `0.06s`):

| Micro-Benchmark | Measured Latency / Ratio |
| :--- | :--- |
| **NF4 Quantize (`131,072` weights, `blocksize=64`)** | **`2.19 ms`** |
| **NF4 Dequantize (`131,072` weights)** | **`0.23 ms`** |
| **NF4 Measured Compression Ratio (vs BF16)** | **`3.56x`** |
| **INT8 Blockwise Roundtrip (`131,072` weights, `blocksize=2048`)** | **`0.59 ms`** |
| **OCP MXFP4 GEMV (`512 x 256`, `E2M1` + `E8M0`)** | **`0.098 ms / iter`** |
| **Fused Chunked Linear Cross-Entropy (`Fwd + Bwd`, vocab=2048, chunk=256)** | **`0.64 ms`** |
| **Release Binary Size (`opt-level=3`, `lto="fat"`, `strip=true`)** | **`752 KB`** |

---

## Quickstart

### 1. Build & Install
```bash
git clone https://github.com/Master-MD/NALA-airllm-rs.git
cd NALA-airllm-rs
./install.sh          # macOS / Linux (or .\install.ps1 on Windows)
```

### 2. Plan a 4 GB GPU Run (Kimi K3 2.8T, Llama-3.1 405B, Qwen3.8-Flash-Next 125B)
```bash
nala-airllm-rs plan --model kimi-k3-2.8t --compression mxfp4 --simulate-vram-gb 4
nala-airllm-rs plan --model qwen3.8-flash-next-125b --compression none --simulate-vram-gb 6
nala-airllm-rs lora-plan --model qwen3.8-27b-vl --rank 16 --seq-len 512 --ce-chunk 4096
```

### 3. Split & Inspect Local Safetensors Checkpoints
```bash
nala-airllm-rs detect /path/to/checkpoint
nala-airllm-rs split --dir /path/to/checkpoint --compression 4bit
nala-airllm-rs inspect /path/to/checkpoint/splitted_model.4bit/model.layers.0.safetensors
```

### 4. Launch the Offline AAA Precision Cockpit & OpenAI/MCP Server
```bash
nala-airllm-rs serve --bind 127.0.0.1:18794
# Open http://127.0.0.1:18794/
```

### 5. Run as a Native MCP Server (12 Tools for Antigravity / Codex / ZCode / OpenCode)
```bash
nala-airllm-rs mcp
```

### 6. Dream / Software-Pearls / REA / RSI Evolution & 2-Click Gate
```bash
# View Solution & Delta Overview Matrix
nala-airllm-rs dream

# Step 1 (safety check — refuses execution without Step 2 confirmation)
nala-airllm-rs dream --apply AIRLLM-RSI-01-DIRECT-NVME-PREFETCH-RING

# Step 2 (creates atomic NCSP SHA-256 snapshot + rollback.sh first)
nala-airllm-rs dream --apply AIRLLM-RSI-01-DIRECT-NVME-PREFETCH-RING --confirm-step2

# Compile a tailored coding prompt for Codex, Antigravity, ZCode, or OpenCode
nala-airllm-rs dream --compile-prompt Antigravity --proposal AIRLLM-RSI-01-DIRECT-NVME-PREFETCH-RING
```

---

## Security & NCSP v2.5 Verification
```bash
RUSTFLAGS="-D warnings" cargo test --release
nala-airllm-rs doctor
nala-airllm-rs audit-pii .
nala-airllm-rs snapshot pre_release
```
- **5-Layer `sHeriFF` Hardening:** `SEC-01` SSRF blockade, `SEC-02` HTML/CSS/SVG injection sanitizer, `SEC-03` Path-traversal & shell metacharacter guard, `SEC-04` 10 MiB body cap + token-bucket rate limiter, `SEC-05` strict security headers + `127.0.0.1` loopback binding.
- **Triple-Council Deliberation Record:** [`docs/TRIPLE_COUNCIL_AIRLLM_RS_20261120.json`](./docs/TRIPLE_COUNCIL_AIRLLM_RS_20261120.json).
