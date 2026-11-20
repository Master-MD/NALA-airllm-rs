# NALA-airllm-rs (`v4.0.0-nala.6.6.6`) — Deutsch (`DE`)

**Sprachen:** **Deutsch (`DE`)** · [English (`EN`)](./README_EN.md) · [Français (`FR`)](./README_FR.md) · [Italiano (`IT`)](./README_IT.md) · [Rumantsch (`RM`)](./README_RM.md) · [5-Sprachen Lizenz](../LICENSE_SUMMARY_5LANG.md) · [60s Caveman Guide](../CAVEMAN_GUIDE.md) · [FAQ](../FAQ.md)

---

**`NALA-airllm-rs`** (`nala-airllm-rs` / Alias `airllm-rs`) ist eine **100 % native Pure-Safe-Rust-Neuimplementierung** (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` externe Crates, Zero-Python, Zero-Node.js, Zero-Telemetry) von [`lyogavin/airllm`](https://github.com/lyogavin/airllm).

Führe **70B- bis 2.8T-Parameter-Modelle auf einer einzelnen 4 GB GPU** aus und trainiere **27B- bis 125B-Modelle per Streamed LoRA unter 6 GB VRAM** — verpackt in ein einziges ~768 KB Binary ohne Python-, PyTorch- oder CUDA-Wheel-Abhängigkeiten.

## Kernfähigkeiten (100 % Parität + Rust-Turbo)
- **Schichtweises Streaming (`embed -> layers[i] -> evict -> norm -> lm_head`):** Hält jeweils nur einen Decoder-Layer im Speicher und überlappt NVMe-I/O via Double-Buffered Prefetch (`sync_channel`, 2 GiB Pinned-RAM Cap).
- **Per-Expert MoE Random-Access Seek:** Liest bei **Kimi K3 (2.8T, 896 Experten, 16 geroutet)** und **DeepSeek-V3 (671B)** nur die tatsächlich gerouteten Experten direkt per Byte-Offset aus `.safetensors` (~1 GB statt 55 GB).
- **Host File-Backed PLE N-Gram Tabellen:** Trennt bei **Qwen3.8-Flash-Next (125B + 51B PLE)** die ~102 GB N-Gram-Tabelle per Longest-Prefix-Owner aus dem Decoder-Layer in eine flache `.mmap`-Datei (0 Bytes GPU-VRAM).
- **Zero-Copy Hardlink Passthrough:** Bereits 1-Layer-pro-Shard vorliegende Checkpoints werden ohne Kopie verlinkt (spart bis zu 1.5 TB Plattenplatz).
- **Pure Rust Block-Weise Kompression:** 4-Bit NF4 (`blocksize=64`, **3.56x** Kompression, `0.23 ms` Dequant), 8-Bit Dynamic (`blocksize=2048`, `0.59 ms`), OCP MXFP4 (`E2M1+E8M0`) GEMV (`0.098 ms/iter`) & FP8 Verbatim-Guard.
- **Streamed Layer-Wise LoRA + Fused Chunked Linear Cross-Entropy:** Trainiert unter 6 GB VRAM ohne `[N, vocab]`-Logits zu materialisieren, inkl. `AdamW` & SFT-Lader (`.jsonl`/`.json`/`.txt` mit `-100` Assistant-Masking).
- **Cross-OS ohne Einschränkungen:** Hebt die macOS Llama/Mistral-Beschränkung von Upstream AirLLM vollständig auf — native Binaries für **macOS (ARM64/Intel)**, **Linux (`x86_64`/`arm64` static musl + `.deb`)** und **Windows (`x86_64`)**.
- **AAA 2-Klick Cockpit, 12-Tool MCP Server & `sHeriFF` Zero-PII Gate:** Offline-Präzisions-Cockpit (`127.0.0.1:18794`), OpenAI-kompatible API, NCSP v2.5 Snapshots (`rollback.sh`) & Multi-IDE Prompt-Compiler (`Codex` / `Antigravity` / `ZCode` / `OpenCode`).

## Unterstützte Frontier-Modelle (Auswahl)
| Modell | Architektur | Parameter | Peak VRAM | Besonderheit |
| :--- | :--- | :--- | :--- | :--- |
| **Kimi K3** | `KimiK3ForConditionalGeneration` | `2.8T` MoE | **`< 4 GB`** | Per-Expert MXFP4 (16/896 Experten) |
| **DeepSeek-V3** | `DeepseekV3ForCausalLM` | `671B` MoE | **`~3.9–11.8 GB`** | FP8 Block-Scale Verbatim + Per-Expert |
| **Llama-3.1 405B** | `LlamaForCausalLM` | `405B` Dense | **`< 4 GB` (4-Bit)** | 126 Layer NF4 Block-Stream |
| **Qwen3-235B-A22B** | `Qwen3MoeForCausalLM` | `235B` MoE | **`~3.1 GB`** | Natives BF16 Streaming |
| **Qwen3.8-Flash-Next** | `Qwen4ExpForConditionalGeneration` | `125B + 51B PLE` | **`5.95 GB`** | 102 GB PLE `.mmap` + Streamed LoRA |
| **Qwen3.8-27B VL** | `Qwen3_5ForConditionalGeneration` | `27B` Dense VL | **`2.0–3.33 GB`** | Gated DeltaNet + LoRA ~2 GB |
| **Llama-3 70B** | `LlamaForCausalLM` | `70B` Dense | **`~2.4 GB`** | Klassisches 4 GB GPU Flaggschiff |

## Schnellstart
```bash
# 1. Aus Release oder lokal bauen & installieren
./install.sh

# 2. 4 GB GPU Roofline-Plan & Live-Benchmark
nala-airllm-rs plan --model kimi-k3-2.8t --compression mxfp4 --simulate-vram-gb 4
nala-airllm-rs bench

# 3. Lokalen Safetensors-Checkpoint splitten
nala-airllm-rs split --dir /pfad/zum/modell --compression 4bit

# 4. Offline AAA 2-Klick Cockpit & MCP Server starten
nala-airllm-rs serve --bind 127.0.0.1:18794
# Browser: http://127.0.0.1:18794/
```
