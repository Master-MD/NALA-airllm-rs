# NALA-airllm-rs (`v4.0.0-nala.6.6.6`) — Français (`FR`)

**`NALA-airllm-rs`** est une réécriture **100 % Rust Pur et Sûr** (`Edition 2024`, `#![forbid(unsafe_code)]`, `0` crate externe, Zéro-Python, Zéro-Télémétrie) de [`lyogavin/airllm`](https://github.com/lyogavin/airllm).

## Points clés
- **Inférence 70B à 2.8T sur un seul GPU 4 Go :** Streaming couche par couche avec préchargement asynchrone double-tampon et éviction immédiate.
- **Streaming MoE par Expert :** Accès aléatoire direct par décalage d'octets dans `.safetensors` (Kimi K3 2.8T : 16 experts sur 896 actifs).
- **Tables PLE N-Gram sur Disque :** Isolation automatique de la table 51B (~102 Go) de Qwen3.8-Flash-Next en fichier `.mmap` hôte sans consommer de VRAM.
- **Compression Pure Rust :** NF4 4-bit, INT8 8-bit, OCP MXFP4 (`E2M1+E8M0`) et protection FP8 verbatim.
- **LoRA en Streaming + Entropie Croisée par Blocs :** Entraînement sous 6 Go VRAM avec chargeur SFT (`.jsonl`, `.json`, `.txt`).
- **Cockpit AAA 2-Clics, Serveur MCP (12 outils) & Porte sHeriFF Zero-PII :** Déploiement natif sur macOS, Linux et Windows avec instantanés NCSP v2.5 (`rollback.sh`).
