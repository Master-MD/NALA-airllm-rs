# NALA-airllm-rs (`v4.0.0-nala.6.6.6`) — Italiano (`IT`)

**Lingue:** [Deutsch (`DE`)](./README_DE.md) · [English (`EN`)](./README_EN.md) · [Français (`FR`)](./README_FR.md) · **Italiano (`IT`)** · [Rumantsch (`RM`)](./README_RM.md) · [Licenza 5 lingue](../LICENSE_SUMMARY_5LANG.md) · [Guida Caveman 60s](../CAVEMAN_GUIDE.md) · [FAQ](../FAQ.md)

---

**`NALA-airllm-rs`** (`nala-airllm-rs` / alias `airllm-rs`) è una riscrittura **100% Pure Safe Rust** (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` crate esterni, Zero-Python, Zero-Node.js, Zero-Telemetria) di [`lyogavin/airllm`](https://github.com/lyogavin/airllm).

Esegui modelli da **70B fino a 2.8T parametri su una singola GPU da 4 GB** e addestra adattatori **LoRA in streaming (27B–125B) sotto i 6 GB di VRAM** — in un singolo binario nativo (~768 KB) per macOS, Linux (musl statico + `.deb`) e Windows.

## Funzionalità principali
- **Streaming strato-per-strato a doppio buffer:** Materializza un solo layer decoder alla volta (`embed -> layers[i] -> evict -> norm -> lm_head`) con prefetch asincrono e limite di 2 GiB di RAM pinned.
- **Streaming MoE per Singolo Esperto (`.safetensors` random-access):** Carica solo gli esperti selezionati dal router tramite byte-offset diretto (Kimi K3 2.8T: 16/896 esperti, ~1 GB invece di 55 GB; DeepSeek-V3 671B).
- **Tabelle PLE N-Gram su Host (`.mmap`):** Separa la tabella da 51B (~102 GB) di Qwen3.8-Flash-Next su disco host con zero byte in VRAM GPU.
- **Hardlink Passthrough Zero-Copy:** Collega direttamente shard già mono-strato senza duplicare terabyte su SSD/NVMe.
- **Compressione Nativa Pure Rust:** NF4 4-bit (**3.56x**, dequant `0.23 ms`), INT8 dinamico (`0.59 ms`), OCP MXFP4 GEMV (`0.098 ms/iter`) e guardia FP8 verbatim.
- **LoRA in Streaming + Chunked Linear Cross-Entropy:** Addestramento sotto i 6 GB VRAM con `AdamW` e caricatore SFT (`.jsonl`, `.json`, `.txt`, mascheramento `-100`).
- **Parità Multi-OS (macOS sbloccato):** Rimuove il limite Llama/Mistral su macOS; binario nativo per macOS (ARM64/Intel), Linux (`x86_64`/`arm64`) e Windows (`x86_64`).
- **Cockpit AAA 2-Click, Server MCP (12 strumenti) & `sHeriFF` Zero-PII:** Binding locale `127.0.0.1:18794`, snapshot atomici NCSP v2.5 (`rollback.sh`) e compilatore prompt Multi-IDE (`Codex` / `Antigravity` / `ZCode` / `OpenCode`).

## Avvio rapido
```bash
./install.sh
nala-airllm-rs plan --model kimi-k3-2.8t --compression mxfp4 --simulate-vram-gb 4
nala-airllm-rs split --dir /percorso/modello --compression 4bit
nala-airllm-rs bench
nala-airllm-rs serve --bind 127.0.0.1:18794
```
