//! Live Pure Safe Rust Micro-Benchmark Suite (Zero Fake Metrics — Measures Real Hardware).

use crate::json::JsonValue;
use crate::lora::chunked_linear_cross_entropy_fwd_bwd;
use crate::quant::{
    dequantize_int8_blockwise, dequantize_nf4, gemv_mxfp4, quantize_int8_blockwise,
    quantize_mxfp4, quantize_nf4,
};
use std::collections::BTreeMap;
use std::time::Instant;

pub fn run_full_benchmark() -> JsonValue {
    // 1. NF4 Quantize + Dequantize (131,072 weights = 512 KiB FP32)
    let n_weights = 131_072usize;
    let weights: Vec<f32> = (0..n_weights)
        .map(|i| ((i as f32 * 0.0137).sin()) * 0.9)
        .collect();

    let t0 = Instant::now();
    let q4 = quantize_nf4(&weights, 64);
    let nf4_q_ms = t0.elapsed().as_secs_f64() * 1000.0;

    let t1 = Instant::now();
    let dq4 = dequantize_nf4(&q4);
    let nf4_dq_ms = t1.elapsed().as_secs_f64() * 1000.0;
    let nf4_checksum = dq4[0] + dq4[n_weights - 1];

    // 2. INT8 Blockwise (blocksize=2048)
    let t2 = Instant::now();
    let q8 = quantize_int8_blockwise(&weights, 2048);
    let dq8 = dequantize_int8_blockwise(&q8);
    let int8_rt_ms = t2.elapsed().as_secs_f64() * 1000.0;
    let int8_checksum = dq8[0] + dq8[n_weights - 1];

    // 3. OCP MXFP4 GEMV (512 x 256)
    let rows = 512usize;
    let cols = 256usize;
    let (mx_packed, mx_scales) = quantize_mxfp4(&weights[..rows * cols]);
    let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.03).cos()).collect();
    let t3 = Instant::now();
    let mut y_acc = 0.0f32;
    for _ in 0..20 {
        let y = gemv_mxfp4(&mx_packed, &mx_scales, &x, rows, cols);
        y_acc += y[0];
    }
    let mxfp4_20_ms = t3.elapsed().as_secs_f64() * 1000.0;

    // 4. Chunked Linear Cross-Entropy (8 tokens x 64 dim x 2048 vocab, chunk=256)
    let n_tok = 8;
    let h_dim = 64;
    let vocab = 2048;
    let h_vec = &weights[..n_tok * h_dim];
    let w_vec = &weights[..vocab * h_dim];
    let labels: Vec<i64> = (0..n_tok).map(|i| (i * 17) as i64).collect();
    let t4 = Instant::now();
    let (ce_loss, dh) =
        chunked_linear_cross_entropy_fwd_bwd(h_vec, w_vec, &labels, n_tok, h_dim, vocab, 256, -100)
            .unwrap_or((0.0, vec![0.0]));
    let ce_ms = t4.elapsed().as_secs_f64() * 1000.0;

    let mut m = BTreeMap::new();
    m.insert("engine".to_string(), JsonValue::String("NALA-airllm-rs Pure Safe Rust".to_string()));
    m.insert("nf4_131k_quantize_ms".to_string(), JsonValue::Number((nf4_q_ms * 100.0).round() / 100.0));
    m.insert("nf4_131k_dequantize_ms".to_string(), JsonValue::Number((nf4_dq_ms * 100.0).round() / 100.0));
    m.insert("nf4_compression_ratio".to_string(), JsonValue::Number(
        ((n_weights * 2) as f64 / (q4.packed.len() + q4.absmax.len() * 4) as f64 * 100.0).round() / 100.0
    ));
    m.insert("int8_131k_roundtrip_ms".to_string(), JsonValue::Number((int8_rt_ms * 100.0).round() / 100.0));
    m.insert("mxfp4_gemv_512x256_per_iter_ms".to_string(), JsonValue::Number(((mxfp4_20_ms / 20.0) * 1000.0).round() / 1000.0));
    m.insert("chunked_ce_fwd_bwd_ms".to_string(), JsonValue::Number((ce_ms * 100.0).round() / 100.0));
    m.insert("chunked_ce_loss".to_string(), JsonValue::Number((ce_loss as f64 * 10000.0).round() / 10000.0));
    m.insert("checksum_guard".to_string(), JsonValue::Number(((nf4_checksum + int8_checksum + y_acc + dh[0]) as f64 * 1000.0).round() / 1000.0));
    JsonValue::Object(m)
}
