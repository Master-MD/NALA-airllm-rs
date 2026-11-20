# NALA-airllm-rs (`v4.0.0-nala.6.6.6`) — Rumantsch (`RM`)

**Linguas:** [Deutsch (`DE`)](./README_DE.md) · [English (`EN`)](./README_EN.md) · [Français (`FR`)](./README_FR.md) · [Italiano (`IT`)](./README_IT.md) · **Rumantsch (`RM`)** · [Licenza 5 linguas](../LICENSE_SUMMARY_5LANG.md) · [Guida Caveman 60s](../CAVEMAN_GUIDE.md) · [FAQ](../FAQ.md)

---

**`NALA-airllm-rs`** (`nala-airllm-rs` / alias `airllm-rs`) è ina nova implementaziun **100 % Pure Safe Rust** (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` crates externs, Zero-Python, Zero-Node.js, Zero-Telemetria) da [`lyogavin/airllm`](https://github.com/lyogavin/airllm).

Lascha ir models da **70B fin 2.8T parameters sin ina suletta GPU da 4 GB** e trenescha **LoRA en streaming (27B–125B) sut 6 GB VRAM** — en in sulet program nativ (~768 KB) per macOS, Linux (musl static + `.deb`) e Windows.

## Funcziuns principalas
- **Streaming stresa per stresa cun double-buffer:** Chargia mo ina suletta stresa a la giada (`embed -> layers[i] -> evict -> norm -> lm_head`) cun prefetch asincron e limita da 2 GiB RAM pinned.
- **MoE Per-Expert Streaming (`.safetensors` random-access):** Legia mo ils experts routads directamain cun byte-offset (Kimi K3 2.8T: 16 da 896 experts, ~1 GB enstagl 55 GB; DeepSeek-V3 671B).
- **Tabellas PLE N-Gram sin il disc (`.mmap`):** Separa la tabella da 51B (~102 GB) da Qwen3.8-Flash-Next sin il disc dal host (0 bytes en la VRAM GPU).
- **Zero-Copy Hardlink Passthrough:** Colligia shards mono-stresa senza duplitgar terabytes sin l’NVMe.
- **Cumpressiun nativa en Pure Rust:** NF4 4-bit (**3.56x**, `0.23 ms` dequant), INT8 dinamic (`0.59 ms`), OCP MXFP4 GEMV (`0.098 ms/iter`) e protecziun FP8 verbatim.
- **Streamed LoRA + Chunked Linear Cross-Entropy:** Trenament sut 6 GB VRAM cun `AdamW` ed SFT-loader (`.jsonl`, `.json`, `.txt`, mascagi `-100`).
- **Paritad Multi-OS (macOS avert cumplettamain):** Nagina restricziun pli sin Llama/Mistral sut macOS; funcziuna nativamain sin macOS (ARM64/Intel), Linux (`x86_64`/`arm64`) e Windows (`x86_64`).
- **Cockpit AAA cun 2 clics, Server MCP (12 utensils) & `sHeriFF` Zero-PII:** Localhost `127.0.0.1:18794`, snapshots atomics NCSP v2.5 (`rollback.sh`) e cumpilatur da prompts Multi-IDE (`Codex` / `Antigravity` / `ZCode` / `OpenCode`).

## Cumenzament svelt
```bash
./install.sh
nala-airllm-rs plan --model kimi-k3-2.8t --compression mxfp4 --simulate-vram-gb 4
nala-airllm-rs split --dir /via/al/model --compression 4bit
nala-airllm-rs bench
nala-airllm-rs serve --bind 127.0.0.1:18794
```
