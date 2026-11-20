#![forbid(unsafe_code)]

//! NALA-airllm-rs — 100% Native Pure Safe Rust (Edition 2024, Zero External Crates, Zero-Python)
//! Layer-Wise & Per-Expert LLM Inference and Streamed LoRA Training Engine.

pub const VERSION: &str = trim_version(include_str!("../VERSION"));

const fn trim_version(s: &'static str) -> &'static str {
    let bytes = s.as_bytes();
    let mut end = bytes.len();
    while end > 0 && (bytes[end - 1] == b'\n' || bytes[end - 1] == b'\r' || bytes[end - 1] == b' ') {
        end -= 1;
    }
    match std::str::from_utf8(bytes.split_at(end).0) {
        Ok(v) => v,
        Err(_) => "4.0.0-nala.6.6.6",
    }
}

pub mod bench;
pub mod catalog;
pub mod dream_rsi;
pub mod hal;
pub mod json;
pub mod lora;
pub mod mcp;
pub mod quant;
pub mod safetensors;
pub mod security;
pub mod server;
pub mod streamer;
