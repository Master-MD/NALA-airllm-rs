# NALA-airllm-rs (`v4.0.0-nala.6.6.6`) — English (`EN`)

**Languages:** [Deutsch (`DE`)](./README_DE.md) · **English (`EN`)** · [Français (`FR`)](./README_FR.md) · [Italiano (`IT`)](./README_IT.md) · [Rumantsch (`RM`)](./README_RM.md) · [5-Language License](../LICENSE_SUMMARY_5LANG.md) · [60s Caveman Guide](../CAVEMAN_GUIDE.md) · [FAQ](../FAQ.md)

---

**`NALA-airllm-rs`** (`nala-airllm-rs` / alias `airllm-rs`) is a **100% Native Pure Safe Rust clean-room rewrite** (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` external crates, Zero-Python, Zero-Node.js, Zero-Telemetry) of [`lyogavin/airllm`](https://github.com/lyogavin/airllm).

Run **70B to 2.8T parameter models on a single 4 GB GPU** and fine-tune **27B to 125B models via Streamed LoRA under 6 GB VRAM** — distributed as standalone native binaries (~768 KB–1.0 MB) for macOS, Linux (static musl + `.deb`), and Windows.

## Key Capabilities
- **Layer-Wise Double-Buffered Streaming:** Materializes one decoder layer at a time (`embed -> layers[i] -> evict -> norm -> lm_head`) with lock-free bounded prefetch (`2 GiB` pinned-RAM safety cap) and microsecond `LayeredProfiler` telemetry.
- **Per-Expert MoE Random-Access Seeking:** Reads only Top-K routed experts by byte offset directly out of `.safetensors` shards (Kimi K3 2.8T: 16 of 896 experts, ~1 GB vs 55 GB; DeepSeek-V3 671B).
- **Host File-Backed PLE N-Gram Tables:** Peels Qwen3.8-Flash-Next's 51B (~102 GB) n-gram table out into a contiguous flat `.mmap` file with direct row-seek gather (`0` bytes in GPU VRAM).
- **Zero-Copy Hardlink Passthrough:** Links pre-sharded single-module checkpoints without duplicating terabytes on NVMe/SSD.
- **Pure Rust Compression Kernels:** 4-bit NF4 (`blocksize=64`, **3.56x** ratio, `0.23 ms` dequant), 8-bit Dynamic (`blocksize=2048`, `0.59 ms`), OCP MXFP4 (`E2M1+E8M0`) GEMV (`0.098 ms/iter`), and FP8 verbatim companion guard.
- **Streamed LoRA + Fused Chunked Linear Cross-Entropy:** Fine-tune under 6 GB VRAM without allocating `[N, vocab]` logits, with `AdamW` and `.jsonl`/`.json`/`.txt` SFT loading (`-100` assistant-only masking).
- **Cross-OS Parity (macOS Restriction Lifted):** Runs every architecture across macOS (Apple Silicon & Intel), Linux (`x86_64` & `arm64`), and Windows (`x86_64`).
- **Hardened OpenAI/MCP Server & Offline AAA 2-Click Cockpit:** `127.0.0.1` loopback binding, `SEC-01..05` + Zero-PII gate, 12 JSON-RPC 2.0 MCP tools, atomic NCSP v2.5 snapshots (`rollback.sh`), and Multi-IDE Prompt Compiler (`Codex` / `Antigravity` / `ZCode` / `OpenCode`).

## Quickstart
```bash
./install.sh
nala-airllm-rs plan --model kimi-k3-2.8t --compression mxfp4 --simulate-vram-gb 4
nala-airllm-rs split --dir /path/to/checkpoint --compression 4bit
nala-airllm-rs bench
nala-airllm-rs serve --bind 127.0.0.1:18794
```
