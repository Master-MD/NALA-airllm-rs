# 60-Second Caveman Guide — `NALA-airllm-rs`

**Big model (70B to 2.8T) + small GPU (4 GB) + Rust (0 Python, 0 Crates) = Fast & Safe.**

## 1. Build (5 seconds)
```bash
cargo build --release
```

## 2. Plan 4 GB GPU Run
```bash
./target/release/nala-airllm-rs plan --model kimi-k3-2.8t --compression mxfp4 --simulate-vram-gb 4
```

## 3. Split Local Safetensors Checkpoint
```bash
./target/release/nala-airllm-rs split --dir /path/to/hf-model --compression 4bit
```
- Already 1-layer-per-shard? Hardlinks in `0 ms` (zero disk duplication).
- Has 51B PLE n-gram table? Peels into `.mmap` on disk (0 bytes in GPU VRAM).

## 4. Open AAA Cockpit (4 GB Roofline + LoRA + 2-Click Dream/RSI)
```bash
./target/release/nala-airllm-rs serve --bind 127.0.0.1:18794
# Open http://127.0.0.1:18794/
```

## 5. Run Live Hardware Benchmark & Zero-PII Check
```bash
./target/release/nala-airllm-rs bench
./target/release/nala-airllm-rs audit-pii .
```

## 6. Connect to Antigravity / Codex / ZCode / OpenCode via MCP
```bash
./target/release/nala-airllm-rs mcp
```
