# NALA-airllm-rs (`v4.0.0-nala.6.6.6`) — Deutsch (`DE`)

**`NALA-airllm-rs`** ist eine zu **100 % native Pure-Safe-Rust-Neuimplementierung** (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` externe Crates, Zero-Python, Zero-Node.js, Zero-Telemetry) von [`lyogavin/airllm`](https://github.com/lyogavin/airllm).

## Kernfähigkeiten (100 % Parität + Rust-Turbo)
- **70B bis 2.8T Modelle auf einer 4 GB GPU:** Schichtweises (`embed -> layers[i] -> evict -> norm -> lm_head`) & expertenweises Streaming direkt von NVMe/SSD.
- **Per-Expert MoE Random-Access Seek:** Liest bei **Kimi K3 (2.8T, 896 Experten, 16 geroutet)** und **DeepSeek-V3 (671B)** nur die tatsächlich gerouteten Experten direkt per Byte-Offset aus `.safetensors` (~1 GB statt 55 GB).
- **Host File-Backed PLE N-Gram Tabellen:** Trennt bei **Qwen3.8-Flash-Next (125B + 51B PLE)** die ~102 GB N-Gram-Tabelle per Longest-Prefix-Owner aus dem Decoder-Layer in eine flache `.mmap`-Datei (0 Bytes GPU-VRAM).
- **Zero-Copy Hardlink Passthrough:** Bereits 1-Layer-pro-Shard vorliegende Checkpoints werden ohne Kopie verlinkt (spart bis zu 1.5 TB Plattenplatz).
- **Block-Weise Kompression in Pure Rust:** 4-Bit NF4 (Dettmers-Codebook, `blocksize=64`, ~3.5x), 8-Bit Dynamic (`blocksize=2048`, ~2.0x), OCP MXFP4 (`E2M1+E8M0`) GEMV & FP8 Verbatim-Guard.
- **Streamed Layer-Wise LoRA + Chunked Linear Cross-Entropy:** Trainiert 27B–125B Modelle unter 6 GB VRAM inkl. `AdamW` & SFT-Lader (`.jsonl`/`.json`/`.txt` mit `-100` Assistant-Masking).
- **Cross-OS ohne Einschränkungen:** Hebt die macOS Llama/Mistral-Beschränkung von Upstream AirLLM vollständig auf — läuft nativ auf macOS (ARM64/Intel), Linux (`x86_64`/`arm64`) und Windows (`x86_64`).
- **AAA Cockpit UI + 12-Tool MCP Server + Dream/REA/RSI 2-Klick-Gate:** Offline-fähiges Präzisions-Cockpit (`http://127.0.0.1:18794/`), NCSP v2.5 Snapshots & `rollback.sh`, `sHeriFF` Zero-PII Gate und 1-Klick Multi-IDE Prompt-Compiler (`Codex` / `Antigravity` / `ZCode` / `OpenCode`).

## Schnellstart
```bash
cargo build --release
./target/release/nala-airllm-rs plan --model kimi-k3-2.8t --compression mxfp4 --simulate-vram-gb 4
./target/release/nala-airllm-rs serve --bind 127.0.0.1:18794
```
