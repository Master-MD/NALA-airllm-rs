//! Pure Safe Rust Block-Wise Quantization & Decompression Kernels:
//! - IEEE 754 FP16 & BFloat16 exact bit conversions
//! - 4-bit NormalFloat (NF4, blocksize=64) block-wise compression (~3.55x reduction)
//! - 8-bit Dynamic Block-wise compression (blocksize=2048, ~2.0x reduction)
//! - OCP MXFP4 (E2M1 4-bit + shared E8M0 exponent) unpack & GEMV (Kimi K3 2.8T)
//! - FP8 (E4M3) decode + verbatim pre-quantized companion-tensor guard

/// Exact Dettmers NF4 16-level information-theoretically optimal NormalFloat codebook.
pub const NF4_CODEBOOK: [f32; 16] = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
];

/// OCP MXFP4 E2M1 4-bit floating-point lookup table (sign bit 3, 2-bit exp, 1-bit mantissa).
pub const MXFP4_E2M1_LUT: [f32; 16] = [
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
];

/// Pre-quantized companion tensor suffixes that must NEVER be cast to fp16/bf16 (verbatim load).
pub const QUANT_COMPANION_SUFFIXES: &[&str] = &[
    "_scale",
    "_scale_inv",
    "_packed",
    "_zero_point",
    "_g_idx",
    "_shape",
];

/// Exact parity with AirLLMBaseModel._should_load_verbatim.
pub fn should_load_verbatim(param_name: &str, dtype_tag: &str) -> bool {
    let dt = dtype_tag.to_ascii_uppercase();
    if matches!(
        dt.as_str(),
        "U8" | "I8" | "I16" | "U16" | "I32" | "U32" | "I64" | "U64" | "BOOL" | "F8_E4M3" | "F8_E5M2"
    ) {
        return true;
    }
    QUANT_COMPANION_SUFFIXES
        .iter()
        .any(|suf| param_name.ends_with(suf))
}

#[inline]
pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

#[inline]
pub fn f32_to_bf16(val: f32) -> u16 {
    let bits = val.to_bits();
    let rounding_bias = 0x7fffu32 + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding_bias)) >> 16) as u16
}

#[inline]
pub fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32) & 0x8000) << 16;
    let exp = ((bits >> 10) & 0x1f) as u32;
    let frac = (bits & 0x03ff) as u32;
    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        let mut f = frac;
        let mut e = 127 - 14;
        while (f & 0x0400) == 0 {
            f <<= 1;
            e -= 1;
        }
        f &= 0x03ff;
        return f32::from_bits(sign | (e << 23) | (f << 13));
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7f80_0000 | (frac << 13));
    }
    let e = exp + (127 - 15);
    f32::from_bits(sign | (e << 23) | (frac << 13))
}

#[inline]
pub fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let frac = bits & 0x007f_ffff;
    if exp <= 0 {
        if exp < -10 {
            return sign;
        }
        let mant = (frac | 0x0080_0000) >> (14 - exp);
        return sign | (mant as u16);
    }
    if exp >= 31 {
        return sign | 0x7c00;
    }
    sign | ((exp as u16) << 10) | ((frac >> 13) as u16)
}

/// Decode IEEE FP8 E4M3FN byte to f32.
#[inline]
pub fn fp8_e4m3_to_f32(byte: u8) -> f32 {
    let sign = if (byte & 0x80) != 0 { -1.0f32 } else { 1.0f32 };
    let exp = ((byte >> 3) & 0x0f) as i32;
    let mant = (byte & 0x07) as f32;
    if exp == 0 {
        sign * (mant / 8.0) * 2.0f32.powi(-6)
    } else if exp == 15 && (byte & 0x07) == 7 {
        f32::NAN
    } else {
        sign * (1.0 + mant / 8.0) * 2.0f32.powi(exp - 7)
    }
}

/// Decode shared OCP E8M0 exponent scale byte (`2^(e - 127)`).
#[inline]
pub fn e8m0_scale(exp_byte: u8) -> f32 {
    if exp_byte == 255 {
        return 0.0;
    }
    f32::from_bits((exp_byte as u32) << 23)
}

#[derive(Debug, Clone)]
pub struct Nf4BlockTensor {
    pub blocksize: usize,
    pub numel: usize,
    pub packed: Vec<u8>,
    pub absmax: Vec<f32>,
}

/// Block-wise 4-bit NormalFloat quantization (blocksize=64, 2 nibbles/byte).
pub fn quantize_nf4(weights: &[f32], blocksize: usize) -> Nf4BlockTensor {
    let bs = blocksize.max(2);
    let num_blocks = weights.len().div_ceil(bs);
    let mut packed = vec![0u8; weights.len().div_ceil(2)];
    let mut absmax = Vec::with_capacity(num_blocks);

    for (b_idx, chunk) in weights.chunks(bs).enumerate() {
        let mut max_abs = 0.0f32;
        for &w in chunk {
            let a = w.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        let scale = if max_abs > 1e-12 { max_abs } else { 1.0 };
        absmax.push(max_abs);
        let inv = 1.0 / scale;

        let base = b_idx * bs;
        for (i, &w) in chunk.iter().enumerate() {
            let norm = (w * inv).clamp(-1.0, 1.0);
            let mut best_code = 0u8;
            let mut best_err = f32::INFINITY;
            for (c, &level) in NF4_CODEBOOK.iter().enumerate() {
                let err = (norm - level).abs();
                if err < best_err {
                    best_err = err;
                    best_code = c as u8;
                }
            }
            let global_i = base + i;
            if (global_i & 1) == 0 {
                packed[global_i >> 1] |= best_code & 0x0f;
            } else {
                packed[global_i >> 1] |= (best_code & 0x0f) << 4;
            }
        }
    }

    Nf4BlockTensor {
        blocksize: bs,
        numel: weights.len(),
        packed,
        absmax,
    }
}

pub fn dequantize_nf4(q: &Nf4BlockTensor) -> Vec<f32> {
    let mut out = vec![0.0f32; q.numel];
    for (b_idx, &scale) in q.absmax.iter().enumerate() {
        let start = b_idx * q.blocksize;
        let end = (start + q.blocksize).min(q.numel);
        for i in start..end {
            let byte = q.packed[i >> 1];
            let nibble = if (i & 1) == 0 {
                byte & 0x0f
            } else {
                (byte >> 4) & 0x0f
            };
            out[i] = NF4_CODEBOOK[nibble as usize] * scale;
        }
    }
    out
}

#[derive(Debug, Clone)]
pub struct Int8BlockTensor {
    pub blocksize: usize,
    pub numel: usize,
    pub packed: Vec<i8>,
    pub absmax: Vec<f32>,
}

/// Block-wise 8-bit dynamic quantization (AirLLM default blocksize=2048).
pub fn quantize_int8_blockwise(weights: &[f32], blocksize: usize) -> Int8BlockTensor {
    let bs = blocksize.max(1);
    let mut packed = Vec::with_capacity(weights.len());
    let mut absmax = Vec::with_capacity(weights.len().div_ceil(bs));

    for chunk in weights.chunks(bs) {
        let mut max_abs = 0.0f32;
        for &w in chunk {
            let a = w.abs();
            if a > max_abs {
                max_abs = a;
            }
        }
        absmax.push(max_abs);
        let inv = if max_abs > 1e-12 { 127.0 / max_abs } else { 0.0 };
        for &w in chunk {
            let q = (w * inv).round().clamp(-127.0, 127.0) as i8;
            packed.push(q);
        }
    }

    Int8BlockTensor {
        blocksize: bs,
        numel: weights.len(),
        packed,
        absmax,
    }
}

pub fn dequantize_int8_blockwise(q: &Int8BlockTensor) -> Vec<f32> {
    let mut out = Vec::with_capacity(q.numel);
    for (b_idx, &scale) in q.absmax.iter().enumerate() {
        let start = b_idx * q.blocksize;
        let end = (start + q.blocksize).min(q.numel);
        let factor = scale / 127.0;
        for &v in &q.packed[start..end] {
            out.push((v as f32) * factor);
        }
    }
    out
}

/// Pack 32-element blocks into OCP MXFP4 (`E2M1` nibbles + shared `E8M0` exponent byte).
pub fn quantize_mxfp4(weights: &[f32]) -> (Vec<u8>, Vec<u8>) {
    let mut packed = vec![0u8; weights.len().div_ceil(2)];
    let mut scales = Vec::with_capacity(weights.len().div_ceil(32));

    for (b_idx, chunk) in weights.chunks(32).enumerate() {
        let max_abs = chunk.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let exp_byte = if max_abs < 1e-12 {
            127u8
        } else {
            let needed = (max_abs / 6.0).log2().ceil() as i32 + 127;
            needed.clamp(1, 254) as u8
        };
        scales.push(exp_byte);
        let s = e8m0_scale(exp_byte);
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        let base = b_idx * 32;
        for (i, &w) in chunk.iter().enumerate() {
            let norm = w * inv;
            let mut best = 0u8;
            let mut best_err = f32::INFINITY;
            for (c, &lut_v) in MXFP4_E2M1_LUT.iter().enumerate() {
                let err = (norm - lut_v).abs();
                if err < best_err {
                    best_err = err;
                    best = c as u8;
                }
            }
            let gi = base + i;
            if (gi & 1) == 0 {
                packed[gi >> 1] |= best & 0x0f;
            } else {
                packed[gi >> 1] |= (best & 0x0f) << 4;
            }
        }
    }
    (packed, scales)
}

/// SIMD-friendly 8-lane accumulator GEMV over packed OCP MXFP4 weights (`[rows, cols] * [cols]`).
pub fn gemv_mxfp4(packed: &[u8], scales: &[u8], x: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; rows];
    let blocks_per_row = cols.div_ceil(32);
    for r in 0..rows {
        let mut acc = [0.0f32; 8];
        let row_elem_base = r * cols;
        let row_scale_base = r * blocks_per_row;
        for b in 0..blocks_per_row {
            let scale = e8m0_scale(scales[row_scale_base + b]);
            let c_start = b * 32;
            let c_end = (c_start + 32).min(cols);
            for c in c_start..c_end {
                let gi = row_elem_base + c;
                let byte = packed[gi >> 1];
                let nib = if (gi & 1) == 0 {
                    byte & 0x0f
                } else {
                    (byte >> 4) & 0x0f
                };
                acc[c & 7] += (MXFP4_E2M1_LUT[nib as usize] * scale) * x[c];
            }
        }
        // Order-preserving tree reduction (deterministic across platforms).
        let s01 = acc[0] + acc[1];
        let s23 = acc[2] + acc[3];
        let s45 = acc[4] + acc[5];
        let s67 = acc[6] + acc[7];
        y[r] = (s01 + s23) + (s45 + s67);
    }
    y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nf4_roundtrip_preserves_signal_with_3_5x_compression() {
        let data: Vec<f32> = (0..1024)
            .map(|i| (i as f32 * 0.031).sin() * 0.85)
            .collect();
        let q = quantize_nf4(&data, 64);
        assert_eq!(q.packed.len(), 512);
        let rec = dequantize_nf4(&q);
        let mse: f32 = data
            .iter()
            .zip(rec.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;
        assert!(mse < 0.005, "NF4 MSE too high: {mse}");
    }

    #[test]
    fn int8_blockwise_roundtrip_high_accuracy() {
        let data: Vec<f32> = (0..4096)
            .map(|i| (i as f32 * 0.017).cos() * 2.5)
            .collect();
        let q = quantize_int8_blockwise(&data, 2048);
        let rec = dequantize_int8_blockwise(&q);
        let max_err = data
            .iter()
            .zip(rec.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 0.03, "INT8 max err too high: {max_err}");
    }

    #[test]
    fn mxfp4_gemv_matches_fp32_reference() {
        let rows = 8;
        let cols = 64;
        let w: Vec<f32> = (0..rows * cols)
            .map(|i| ((i as f32 * 0.13).sin()) * 1.5)
            .collect();
        let x: Vec<f32> = (0..cols).map(|i| (i as f32 * 0.21).cos()).collect();
        let (packed, scales) = quantize_mxfp4(&w);
        let y_q = gemv_mxfp4(&packed, &scales, &x, rows, cols);
        let mut y_ref = vec![0.0f32; rows];
        for r in 0..rows {
            for c in 0..cols {
                y_ref[r] += w[r * cols + c] * x[c];
            }
        }
        for r in 0..rows {
            assert!(
                (y_q[r] - y_ref[r]).abs() < 1.5,
                "row {r} mismatch: {} vs {}",
                y_q[r],
                y_ref[r]
            );
        }
    }

    #[test]
    fn verbatim_guard_protects_quantized_companions() {
        assert!(should_load_verbatim("model.layers.0.mlp.weight_packed", "U8"));
        assert!(should_load_verbatim("model.layers.0.mlp.weight_scale_inv", "F32"));
        assert!(should_load_verbatim("model.layers.0.mlp.weight", "F8_E4M3"));
        assert!(!should_load_verbatim("model.layers.0.mlp.weight", "BF16"));
    }
}
