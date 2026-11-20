#!/usr/bin/env bash
set -euo pipefail

# NALA-airllm-rs Universal Installer (macOS ARM64/Intel & Linux x86_64/ARM64)
REPO_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="${NALA_BIN_DIR:-$HOME/bin}"

echo "[NALA-airllm-rs] Building 100% Pure Safe Rust release binary..."
cargo build --release --manifest-path "$REPO_DIR/Cargo.toml"

mkdir -p "$INSTALL_DIR"
cp "$REPO_DIR/target/release/nala-airllm-rs" "$INSTALL_DIR/nala-airllm-rs"
chmod +x "$INSTALL_DIR/nala-airllm-rs"
ln -sf "$INSTALL_DIR/nala-airllm-rs" "$INSTALL_DIR/airllm-rs"

echo "[NALA-airllm-rs] Installed:"
echo "  - $INSTALL_DIR/nala-airllm-rs"
echo "  - $INSTALL_DIR/airllm-rs (symlink)"
"$INSTALL_DIR/nala-airllm-rs" version
