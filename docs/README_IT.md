# NALA-airllm-rs (`v4.0.0-nala.6.6.6`) — Italiano (`IT`)

**`NALA-airllm-rs`** è una riscrittura **100% Pure Safe Rust** (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` crate esterni, Zero-Python, Zero-Telemetria) di [`lyogavin/airllm`](https://github.com/lyogavin/airllm).

## Caratteristiche principali
- **Modelli da 70B a 2.8T su una singola GPU da 4 GB:** Streaming strato-per-strato con prefetch asincrono a doppio buffer e rilascio immediato della VRAM.
- **Streaming MoE per Singolo Esperto:** Lettura diretta tramite offset nei file `.safetensors` (Kimi K3 2.8T: 16/896 esperti attivi).
- **Tabelle PLE N-Gram su Host (`.mmap`):** Estrazione automatica della tabella 51B (~102 GB) di Qwen3.8-Flash-Next su disco host (0 byte in VRAM).
- **Compressione Nativa Rust:** NF4 4-bit, INT8 8-bit, OCP MXFP4 (`E2M1+E8M0`) e guardia FP8 verbatim.
- **LoRA in Streaming + Chunked Cross-Entropy:** Addestramento sotto i 6 GB VRAM con caricatore SFT (`.jsonl`, `.json`, `.txt`).
- **Cockpit AAA 2-Click, Server MCP (12 strumenti) & sHeriFF Zero-PII:** Esecuzione universale su macOS, Linux e Windows con snapshot atomici NCSP v2.5 (`rollback.sh`).
