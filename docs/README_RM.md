# NALA-airllm-rs (`v4.0.0-nala.6.6.6`) — Rumantsch (`RM`)

**`NALA-airllm-rs`** è ina nova implementaziun **100 % Pure Safe Rust** (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` crates externs, Zero-Python, Zero-Telemetria) da [`lyogavin/airllm`](https://github.com/lyogavin/airllm).

## Funcziuns principalas
- **Models da 70B fin 2.8T sin ina GPU da 4 GB:** Streaming stresa per stresa cun prefetch asincron ed evicziun immediata da la VRAM.
- **MoE Per-Expert Streaming:** Lectura directa dals experts routads ord datotecas `.safetensors` (Kimi K3 2.8T, DeepSeek-V3 671B).
- **Tabellas PLE N-Gram sin il disc (`.mmap`):** Separaziun da la tabella da 102 GB da Qwen3.8-Flash-Next (0 bytes en la VRAM).
- **Cumpressiun nativa en Rust:** NF4 4-bit, INT8 8-bit, OCP MXFP4 (`E2M1+E8M0`) e protecziun FP8.
- **LoRA en streaming + Chunked Cross-Entropy:** Trenament sut 6 GB VRAM cun SFT (`.jsonl`, `.json`, `.txt`).
- **Cockpit AAA cun 2 clics, Server MCP (12 utensils) & sHeriFF Zero-PII:** Funcziuna sin macOS, Linux e Windows cun snapshots NCSP v2.5 (`rollback.sh`).
