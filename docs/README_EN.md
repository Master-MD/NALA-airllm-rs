# NALA-airllm-rs (`v4.0.0-nala.6.6.6`) — English (`EN`)

**`NALA-airllm-rs`** is a **100% Native Pure Safe Rust clean-room rewrite** (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` external crates, Zero-Python, Zero-Node.js, Zero-Telemetry) of [`lyogavin/airllm`](https://github.com/lyogavin/airllm).

## Key Capabilities
- **Run 70B to 2.8T Models on a Single 4 GB GPU:** Double-buffered layer-wise streaming (`embed -> layer[i] -> evict -> norm -> lm_head`) with deterministic VRAM eviction.
- **Per-Expert MoE Random-Access Seeking:** Streams only Top-K routed experts by byte offset from `.safetensors` shards (Kimi K3 2.8T: 16/896 experts, ~1 GB vs 55 GB).
- **Host File-Backed PLE N-Gram Tables:** Peels Qwen3.8-Flash-Next's 51B (~102 GB) n-gram table out into a contiguous flat `.mmap` file with direct row-seek gather (0 bytes in GPU VRAM).
- **Zero-Copy Hardlink Passthrough:** Links pre-sharded single-module checkpoints without duplicating terabytes of weights.
- **Pure Rust Compression Kernels:** 4-bit NF4 (`blocksize=64`, ~3.5x), 8-bit Dynamic (`blocksize=2048`, ~2x), OCP MXFP4 (`E2M1+E8M0`) GEMV, and FP8 verbatim companion protection.
- **Streamed LoRA + Fused Chunked Linear Cross-Entropy:** Fine-tune 27B–125B models under 6 GB VRAM without materializing `[N, vocab]` logits, with `.jsonl`/`.json`/`.txt` SFT loading.
- **Cross-OS Parity:** Lifts upstream macOS Llama/Mistral restrictions; runs natively on macOS (ARM64 & Intel), Linux (`x86_64` & `arm64`), and Windows (`x86_64`).
- **Hardened OpenAI/MCP Server & AAA 2-Click Dream/REA/RSI Cockpit:** `127.0.0.1` localhost binding, 12 JSON-RPC 2.0 MCP tools, atomic NCSP v2.5 snapshots + `rollback.sh`, `sHeriFF` Zero-PII gate, and Multi-IDE Prompt Compiler (`Codex` / `Antigravity` / `ZCode` / `OpenCode`).
