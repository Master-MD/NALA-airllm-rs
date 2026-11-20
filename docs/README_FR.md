# NALA-airllm-rs (`v4.0.0-nala.6.6.6`) — Français (`FR`)

**Langues :** [Deutsch (`DE`)](./README_DE.md) · [English (`EN`)](./README_EN.md) · **Français (`FR`)** · [Italiano (`IT`)](./README_IT.md) · [Rumantsch (`RM`)](./README_RM.md) · [Licence 5 langues](../LICENSE_SUMMARY_5LANG.md) · [Guide Caveman 60s](../CAVEMAN_GUIDE.md) · [FAQ](../FAQ.md)

---

**`NALA-airllm-rs`** (`nala-airllm-rs` / alias `airllm-rs`) est une réécriture **100 % Rust Pur et Sûr** (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` crate externe, Zéro-Python, Zéro-Node.js, Zéro-Télémétrie) de [`lyogavin/airllm`](https://github.com/lyogavin/airllm).

Exécutez des modèles de **70B à 2.8T paramètres sur un seul GPU de 4 Go** et entraînez des adaptateurs **LoRA en streaming (27B à 125B) sous 6 Go de VRAM** — dans un binaire natif autonome (~768 Ko) pour macOS, Linux (musl statique + `.deb`) et Windows.

## Capacités clés
- **Streaming couche par couche double-tampon :** Ne charge qu’une seule couche décodeur à la fois (`embed -> layers[i] -> evict -> norm -> lm_head`) avec préchargement asynchrone et limite de RAM épinglée de 2 Gio.
- **Streaming MoE par Expert (Accès direct `.safetensors`) :** Lit uniquement les experts routés par décalage d’octets (Kimi K3 2.8T : 16/896 experts, ~1 Go au lieu de 55 Go ; DeepSeek-V3 671B).
- **Tables PLE N-Gram sur Disque (`.mmap`) :** Extrait la table 51B (~102 Go) de Qwen3.8-Flash-Next en fichier plat hôte sans jamais consommer de VRAM GPU.
- **Hardlink Passthrough Zéro-Copie :** Relie directement les shards mono-couche sans dupliquer des téraoctets sur NVMe.
- **Compression Pure Rust :** NF4 4-bit (**3.56x**, déquantification `0.23 ms`), INT8 dynamique (`0.59 ms`), GEMV OCP MXFP4 (`0.098 ms/iter`) et protection FP8 verbatim.
- **LoRA en Streaming + Entropie Croisée par Blocs :** Entraînement sous 6 Go VRAM avec `AdamW` et chargeur SFT (`.jsonl`, `.json`, `.txt`, masquage assistant `-100`).
- **Parité Multi-OS (macOS débloqué) :** Lève la restriction Llama/Mistral sur macOS ; fonctionne nativement sur macOS (ARM64/Intel), Linux (`x86_64`/`arm64`) et Windows (`x86_64`).
- **Cockpit AAA 2-Clics, Serveur MCP (12 outils) & Porte `sHeriFF` Zéro-PII :** Liaison stricte `127.0.0.1:18794`, snapshots NCSP v2.5 (`rollback.sh`) et compilateur de prompts Multi-IDE (`Codex` / `Antigravity` / `ZCode` / `OpenCode`).

## Démarrage rapide
```bash
./install.sh
nala-airllm-rs plan --model kimi-k3-2.8t --compression mxfp4 --simulate-vram-gb 4
nala-airllm-rs split --dir /chemin/modele --compression 4bit
nala-airllm-rs bench
nala-airllm-rs serve --bind 127.0.0.1:18794
```
