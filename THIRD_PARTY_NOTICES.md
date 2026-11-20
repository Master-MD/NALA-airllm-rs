# Third-Party Notices

## 1. Upstream AirLLM (`lyogavin/airllm`)
- **Copyright:** Gavin Li (`lyogavin`) & AirLLM Contributors
- **License:** Apache License 2.0 (included verbatim in [`LICENSE`](./LICENSE))
- **Preserved Reference:** Original upstream documentation is preserved in [`README_UPSTREAM_AIRLLM.md`](./README_UPSTREAM_AIRLLM.md).
- **Changes Made in `NALA-airllm-rs` (`v4.0.0-nala.6.6.6`):**
  - Complete clean-room rewrite from Python/PyTorch/bitsandbytes/MLX into 100% Native Pure Safe Rust (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` external crates, Zero-Python, Zero-Node.js).
  - Full feature parity across `AutoModel` architecture detection, 10+ model layout overrides (`Kimi K3 2.8T`, `DeepSeek-V3 671B`, `Llama-3.1 405B`, `Qwen3-235B`, `Qwen3.8-Flash-Next 125B + 51B PLE`, `Qwen3.8-27B`, `Mixtral`, `ChatGLM`, `Baichuan`, `InternLM`), zero-copy hardlink passthrough, random-access per-expert Safetensors seeking, contiguous PLE n-gram raw `.mmap` peeling & file-seek gather, NF4/INT8/MXFP4 block-wise compression, double-buffered prefetching, Streamed LoRA + Fused Chunked Linear Cross-Entropy, and SFT `.jsonl`/`.json`/`.txt` loading.
  - Added Cross-OS NALA-HAL (lifting macOS Llama/Mistral-only restriction), `sHeriFF` 5-layer security + Zero-PII gate, JSON-RPC 2.0 MCP Server (12 tools), OpenAI-compatible localhost HTTP server, offline AAA Swiss-Precision Cockpit UI, and 2-Click Dream/REA/RSI Evolution & Multi-IDE Prompt Studio.

## 2. Rust Standard Library
- **License:** Dual-licensed under Apache-2.0 / MIT.
- **External Crate Dependencies:** `0` (`Cargo.toml` has zero third-party crate dependencies).
