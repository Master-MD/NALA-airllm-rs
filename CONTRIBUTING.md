# Contributing to `NALA-airllm-rs`

Thank you for contributing to `NALA-airllm-rs`! To preserve our **Pure Safe Rust**, **Zero-Regression**, and **Zero-PII** guarantees, every Pull Request must satisfy:

1. **Zero External Crates & Zero Unsafe Code:** Keep `#![forbid(unsafe_code)]` and `0` third-party dependencies in `Cargo.toml`.
2. **Zero Compiler Warnings & 100% Test Pass:**
   ```bash
   RUSTFLAGS="-D warnings" cargo test --release
   RUSTFLAGS="-D warnings" cargo build --release
   ```
3. **Zero-PII & Security Gate:**
   ```bash
   ./target/release/nala-airllm-rs audit-pii .
   ./target/release/nala-airllm-rs doctor
   ```
4. **Version SSOT:** `VERSION` is the single source of truth; do not hardcode mismatched version strings.
5. **Apache-2.0 Licensing:** All contributions are licensed under Apache License 2.0 (see [`LICENSE`](./LICENSE)).
