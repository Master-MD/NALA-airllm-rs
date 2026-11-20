# Frequently Asked Questions (`NALA-airllm-rs`)

## 1. Does `NALA-airllm-rs` still require Python, PyTorch, `transformers`, `bitsandbytes`, or `mlx`?
**No.** `NALA-airllm-rs v4.0.0-nala.6.6.6` is compiled from **100% Pure Safe Rust (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` external crates)**. It runs as a single native binary (~600 KB) on macOS (Apple Silicon & Intel), Linux (`x86_64` & `arm64`), and Windows (`x86_64`) with zero runtime dependencies.

## 2. How does it run 70B–2.8T models on a 4 GB GPU?
1. **Layer-Wise Streaming:** Only one transformer layer is materialized at a time (`embed -> layer[i] -> evict -> layer[i+1]`).
2. **Per-Expert MoE Random-Access Seek:** On sparse MoE models like **Kimi K3 (2.8T, 896 experts, 16 routed)**, `nala-airllm-rs` seeks directly to only the routed experts' byte offsets inside `.safetensors` shards (~1 GB active instead of 55 GB).
3. **Host File-Backed PLE Tables:** On **Qwen3.8-Flash-Next (125B + 51B PLE)**, the ~102 GB n-gram table is peeled out into a flat `.mmap` file and gathered by row seek on the host without ever touching GPU VRAM.
4. **Block-Wise Compression:** Pure Rust NF4 (`4bit`, 3.55x reduction), INT8 (`8bit`, 2.0x), and OCP MXFP4 (`E2M1+E8M0`) slash NVMe-to-memory transfer times.

## 3. Can I train LoRA adapters on 27B–125B models under 6 GB VRAM?
Yes. `src/lora.rs` implements **Streamed Layer-Wise LoRA** (`LoRALinear` + `AdamW` + `.safetensors` adapter save/load), **Fused Chunked Linear Cross-Entropy** (online `logaddexp` across vocab slices so `[N, 248320]` logits are never materialized), and a full SFT loader (`.jsonl`, `.json`, `.txt` with automatic `-100` assistant-only masking).

## 4. Does macOS still restrict me to Llama/Mistral models?
**No.** Upstream AirLLM restricted macOS to `LlamaForCausalLM` and `MistralForCausalLM` due to its MLX port. `NALA-airllm-rs` lifts that restriction: all architectures run natively on macOS ARM64 & Intel via the unified Pure Rust engine.

## 5. How does the 2-Click Dream / REA / RSI Evolution Gate work?
All self-evolution proposals (evaluated over 3–7 rounds by `Grand NALAtheON`, `Grand NEOP`, and `Grand TheGremium / TheForum`) are presented in a Solution & Delta Overview Matrix (`Δ Performance`, `Δ Features`, `Δ Sicherheit`). Nothing executes without **2 explicit user clicks** (`1. Klick: Installieren/Umsetzen` -> `"Wollen Sie wirklich?"` -> `2. Klick: AUSFÜHREN`), and every confirmed execution automatically seals an atomic **NCSP v2.5 SHA-256 snapshot + `rollback.sh`** first.

## 6. Is any telemetry or private data sent anywhere?
**Zero (`0%`).** The HTTP/Cockpit server binds strictly to `127.0.0.1`, enforces `SEC-01..SEC-05` protections, and every commit passes the `sHeriFF` Zero-PII gate (`nala-airllm-rs audit-pii`).
