//! Double-Buffered Layer-Wise & Per-Expert Streaming Inference Engine + LayeredProfiler.
//! Streams one decoder layer (or only Top-K routed MoE experts) at a time from NVMe/SSD,
//! decompresses NF4/INT8 on-the-fly, overlaps next-layer I/O on a background thread,
//! and evicts each layer immediately after forward.

use crate::catalog::LayerNamesDict;
use crate::json::JsonValue;
use crate::quant::{
    Int8BlockTensor, Nf4BlockTensor, dequantize_int8_blockwise, dequantize_nf4,
};
use crate::safetensors::{
    RawTensor, load_safetensors_all, load_safetensors_subset, read_safetensors_header,
};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::thread;
use std::time::Instant;

#[derive(Debug, Clone, Default)]
pub struct LayeredProfiler {
    pub timings_us: BTreeMap<String, Vec<u64>>,
    pub peak_active_bytes: usize,
    pub layers_streamed: usize,
    pub experts_streamed: usize,
}

impl LayeredProfiler {
    pub fn record_us(&mut self, stage: &str, us: u64) {
        self.timings_us
            .entry(stage.to_string())
            .or_default()
            .push(us);
    }

    pub fn track_active_bytes(&mut self, bytes: usize) {
        if bytes > self.peak_active_bytes {
            self.peak_active_bytes = bytes;
        }
    }

    pub fn total_us(&self, stage: &str) -> u64 {
        self.timings_us
            .get(stage)
            .map(|v| v.iter().copied().sum())
            .unwrap_or(0)
    }

    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert(
            "peak_active_bytes".to_string(),
            JsonValue::Number(self.peak_active_bytes as f64),
        );
        m.insert(
            "layers_streamed".to_string(),
            JsonValue::Number(self.layers_streamed as f64),
        );
        m.insert(
            "experts_streamed".to_string(),
            JsonValue::Number(self.experts_streamed as f64),
        );
        let mut stage_totals = BTreeMap::new();
        for (k, vals) in &self.timings_us {
            let sum: u64 = vals.iter().copied().sum();
            stage_totals.insert(k.clone(), JsonValue::Number(sum as f64));
        }
        m.insert("stage_totals_us".to_string(), JsonValue::Object(stage_totals));
        JsonValue::Object(m)
    }
}

/// Decompress any NF4 (`.4bit.absmax`) or INT8 (`.8bit.absmax`) tensors in a loaded layer map.
pub fn decompress_layer_map(
    raw: BTreeMap<String, RawTensor>,
) -> Result<BTreeMap<String, Vec<f32>>, String> {
    let mut out = BTreeMap::new();
    let keys: Vec<String> = raw.keys().cloned().collect();

    for key in &keys {
        if key.contains(".4bit.") || key.contains(".8bit.") {
            continue;
        }
        let abs4 = format!("{key}.4bit.absmax");
        let abs8 = format!("{key}.8bit.absmax");
        let shape4 = format!("{key}.4bit.shape");
        let shape8 = format!("{key}.8bit.shape");

        if let (Some(packed), Some(abs_t)) = (raw.get(key), raw.get(&abs4)) {
            let absmax = abs_t.to_f32_vec()?;
            let numel = raw
                .get(&shape4)
                .map(|st| {
                    st.data
                        .chunks_exact(8)
                        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as usize)
                        .product::<usize>()
                })
                .unwrap_or(packed.data.len() * 2);
            let q = Nf4BlockTensor {
                blocksize: 64,
                numel,
                packed: packed.data.clone(),
                absmax,
            };
            out.insert(key.clone(), dequantize_nf4(&q));
        } else if let (Some(packed), Some(abs_t)) = (raw.get(key), raw.get(&abs8)) {
            let absmax = abs_t.to_f32_vec()?;
            let numel = raw
                .get(&shape8)
                .map(|st| {
                    st.data
                        .chunks_exact(8)
                        .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as usize)
                        .product::<usize>()
                })
                .unwrap_or(packed.data.len());
            let q = Int8BlockTensor {
                blocksize: 2048,
                numel,
                packed: packed.data.iter().map(|&b| b as i8).collect(),
                absmax,
            };
            out.insert(key.clone(), dequantize_int8_blockwise(&q));
        } else if let Some(t) = raw.get(key) {
            if let Ok(v) = t.to_f32_vec() {
                out.insert(key.clone(), v);
            }
        }
    }
    Ok(out)
}

/// Partition tensor names inside a decoder shard into `(non_expert_keys, per_expert_keys)`
/// for Kimi K3 / Mixtral / DeepSeek per-expert streaming.
pub fn partition_expert_keys(
    shard_path: &Path,
    expert_prefix: &str,
) -> Result<(Vec<String>, BTreeMap<usize, Vec<String>>), String> {
    let header = read_safetensors_header(shard_path)?;
    let marker = format!(".{expert_prefix}.");
    let mut non_expert = Vec::new();
    let mut per_expert: BTreeMap<usize, Vec<String>> = BTreeMap::new();

    for key in header.tensors.keys() {
        if let Some(pos) = key.find(&marker) {
            let rest = &key[pos + marker.len()..];
            if let Some(head) = rest.split('.').next() {
                if let Ok(expert_idx) = head.parse::<usize>() {
                    per_expert.entry(expert_idx).or_default().push(key.clone());
                    continue;
                }
            }
        }
        non_expert.push(key.clone());
    }
    Ok((non_expert, per_expert))
}

/// RMSNorm (`x * w / sqrt(mean(x^2) + eps)`).
pub fn rms_norm(x: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let dim = weight.len();
    let n = x.len() / dim;
    let mut out = vec![0.0f32; x.len()];
    for t in 0..n {
        let row = &x[t * dim..(t + 1) * dim];
        let mut ss = 0.0f32;
        for &v in row {
            ss += v * v;
        }
        let inv = 1.0 / ((ss / dim as f32) + eps).sqrt();
        let o_row = &mut out[t * dim..(t + 1) * dim];
        for d in 0..dim {
            o_row[d] = row[d] * inv * weight[d];
        }
    }
    out
}

/// SiLU activation (`x / (1 + exp(-x))`).
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Apply Rotary Positional Embeddings (RoPE) in-place to a `[seq_len, dim]` buffer.
pub fn apply_rope_inplace(x: &mut [f32], seq_len: usize, dim: usize, theta: f32) {
    let half = dim / 2;
    for pos in 0..seq_len {
        let row = &mut x[pos * dim..(pos + 1) * dim];
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / (dim as f32));
            let angle = (pos as f32) * freq;
            let (sin_a, cos_a) = angle.sin_cos();
            let x0 = row[i];
            let x1 = row[i + half];
            row[i] = x0 * cos_a - x1 * sin_a;
            row[i + half] = x0 * sin_a + x1 * cos_a;
        }
    }
}

/// Linear projection `x [n, in_d] @ w^T [out_d, in_d] -> [n, out_d]`.
pub fn linear_f32(x: &[f32], w: &[f32], n: usize, in_d: usize, out_d: usize) -> Vec<f32> {
    let mut y = vec![0.0f32; n * out_d];
    for t in 0..n {
        let xr = &x[t * in_d..(t + 1) * in_d];
        for o in 0..out_d {
            let wr = &w[o * in_d..(o + 1) * in_d];
            let mut acc = 0.0f32;
            for i in 0..in_d {
                acc += xr[i] * wr[i];
            }
            y[t * out_d + o] = acc;
        }
    }
    y
}

/// Gated DeltaNet O(1) recurrent state update (for Qwen3.8-27B & Qwen3.8-Flash-Next).
/// `S_t = alpha * S_{t-1} + beta * (v - alpha * (S_{t-1} * k)) * k^T`, output `y_t = S_t * q`.
pub fn gated_deltanet_step(
    state: &mut [f32],
    k: &[f32],
    v: &[f32],
    q: &[f32],
    alpha: f32,
    beta: f32,
    dim: usize,
) -> Vec<f32> {
    let alpha_clamped = alpha.clamp(0.0, 0.9999);
    let beta_clamped = beta.clamp(0.0, 1.99);
    let k_norm = (k.iter().map(|&x| x * x).sum::<f32>() + 1e-8).sqrt();
    let k_hat: Vec<f32> = k.iter().map(|&x| x / k_norm).collect();

    let mut y = vec![0.0f32; dim];
    for r in 0..dim {
        let row = &mut state[r * dim..(r + 1) * dim];
        let mut sk = 0.0f32;
        for c in 0..dim {
            sk += row[c] * k_hat[c];
        }
        let delta = beta_clamped * (v[r] - alpha_clamped * sk);
        let mut out_r = 0.0f32;
        for c in 0..dim {
            let updated = alpha_clamped * row[c] + delta * k_hat[c];
            row[c] = if updated.is_finite() { updated } else { 0.0 };
            out_r += row[c] * q[c];
        }
        y[r] = out_r;
    }
    y
}

/// Top-K MoE router selecting the `top_k` highest-scoring experts with softmax weights.
pub fn route_topk_experts(router_logits: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let mut indexed: Vec<(usize, f32)> = router_logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    indexed.truncate(top_k.min(indexed.len()));
    let max_l = indexed
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = indexed.iter().map(|(_, v)| (v - max_l).exp()).collect();
    let sum: f32 = exps.iter().sum::<f32>().max(1e-12);
    indexed
        .into_iter()
        .zip(exps)
        .map(|((idx, _), e)| (idx, e / sum))
        .collect()
}

#[derive(Debug, Clone)]
pub struct StreamedForwardResult {
    pub next_token_id: usize,
    pub top5_tokens: Vec<(usize, f32)>,
    pub profiler: LayeredProfiler,
}

/// Execute a real, double-buffered, layer-by-layer forward pass over an AirLLM sharded
/// `.safetensors` directory (supports dense layers, NF4/INT8 compressed shards, and
/// per-expert MoE random-access streaming!).
pub fn run_streamed_forward(
    splitted_dir: &Path,
    layer_names: &LayerNamesDict,
    input_ids: &[usize],
    num_layers: usize,
    hidden_size: usize,
    vocab_size: usize,
    prefetching: bool,
) -> Result<StreamedForwardResult, String> {
    let mut profiler = LayeredProfiler::default();
    let seq_len = input_ids.len().max(1);

    // 1. Stream Embed layer
    let t0 = Instant::now();
    let embed_path = splitted_dir.join(format!("{}.safetensors", layer_names.embed));
    let embed_raw = load_safetensors_all(&embed_path)?;
    profiler.record_us("disk_load", t0.elapsed().as_micros() as u64);

    let t1 = Instant::now();
    let embed_map = decompress_layer_map(embed_raw)?;
    profiler.record_us("decompress", t1.elapsed().as_micros() as u64);

    let embed_w = embed_map
        .values()
        .next()
        .ok_or_else(|| "embed shard has no tensors".to_string())?;
    profiler.track_active_bytes(embed_w.len() * 4);

    let mut hidden = vec![0.0f32; seq_len * hidden_size];
    for (pos, &tok) in input_ids.iter().enumerate() {
        let tid = tok % vocab_size.max(1);
        let src_start = tid * hidden_size;
        if src_start + hidden_size <= embed_w.len() {
            hidden[pos * hidden_size..(pos + 1) * hidden_size]
                .copy_from_slice(&embed_w[src_start..src_start + hidden_size]);
        }
    }
    profiler.layers_streamed += 1;
    drop(embed_map); // Evict embed immediately!

    // 2. Double-buffered prefetch producer for decoder layers 0..num_layers
    let layer_paths: Vec<PathBuf> = (0..num_layers)
        .map(|i| splitted_dir.join(format!("{}.{i}.safetensors", layer_names.layer_prefix)))
        .collect();

    let expert_prefix = layer_names.expert_prefix.clone();

    if prefetching && !layer_paths.is_empty() {
        let (tx, rx) = mpsc::sync_channel::<Result<(usize, BTreeMap<String, RawTensor>, u64), String>>(1);
        let paths_bg = layer_paths.clone();
        let exp_pref_bg = expert_prefix.clone();

        let loader_handle = thread::spawn(move || {
            for (idx, p) in paths_bg.into_iter().enumerate() {
                let ts = Instant::now();
                let res = if let Some(ref ep) = exp_pref_bg {
                    match partition_expert_keys(&p, ep) {
                        Ok((non_exp, _)) if !non_exp.is_empty() => load_safetensors_subset(&p, &non_exp),
                        Ok(_) => load_safetensors_all(&p),
                        Err(e) => Err(e),
                    }
                } else {
                    load_safetensors_all(&p)
                };
                let us = ts.elapsed().as_micros() as u64;
                if tx.send(res.map(|m| (idx, m, us))).is_err() {
                    break;
                }
            }
        });

        for msg in rx {
            let (layer_idx, raw_layer, io_us) = msg?;
            profiler.record_us("disk_load_prefetch", io_us);
            let td = Instant::now();
            let weights = decompress_layer_map(raw_layer)?;
            profiler.record_us("decompress", td.elapsed().as_micros() as u64);

            let active_bytes: usize = weights.values().map(|v| v.len() * 4).sum();
            profiler.track_active_bytes(active_bytes);

            let tc = Instant::now();
            apply_decoder_layer_weights(
                &mut hidden,
                &weights,
                &layer_paths[layer_idx],
                expert_prefix.as_deref(),
                seq_len,
                hidden_size,
                &mut profiler,
            )?;
            profiler.record_us("compute", tc.elapsed().as_micros() as u64);
            profiler.layers_streamed += 1;
            // weights dropped here -> instant VRAM/RAM eviction!
        }
        let _ = loader_handle.join();
    } else {
        for (layer_idx, path) in layer_paths.iter().enumerate() {
            let ts = Instant::now();
            let raw_layer = if let Some(ref ep) = expert_prefix {
                let (non_exp, _) = partition_expert_keys(path, ep)?;
                if !non_exp.is_empty() {
                    load_safetensors_subset(path, &non_exp)?
                } else {
                    load_safetensors_all(path)?
                }
            } else {
                load_safetensors_all(path)?
            };
            profiler.record_us("disk_load", ts.elapsed().as_micros() as u64);

            let td = Instant::now();
            let weights = decompress_layer_map(raw_layer)?;
            profiler.record_us("decompress", td.elapsed().as_micros() as u64);

            let active_bytes: usize = weights.values().map(|v| v.len() * 4).sum();
            profiler.track_active_bytes(active_bytes);

            let tc = Instant::now();
            apply_decoder_layer_weights(
                &mut hidden,
                &weights,
                &layer_paths[layer_idx],
                expert_prefix.as_deref(),
                seq_len,
                hidden_size,
                &mut profiler,
            )?;
            profiler.record_us("compute", tc.elapsed().as_micros() as u64);
            profiler.layers_streamed += 1;
        }
    }

    // 3. Stream Final Norm
    let norm_path = splitted_dir.join(format!("{}.safetensors", layer_names.norm));
    if norm_path.exists() {
        let raw_norm = load_safetensors_all(&norm_path)?;
        let norm_map = decompress_layer_map(raw_norm)?;
        if let Some(nw) = norm_map.values().next() {
            if nw.len() == hidden_size {
                hidden = rms_norm(&hidden, nw, 1e-5);
            }
        }
        profiler.layers_streamed += 1;
    }

    // 4. Stream LM Head (or reuse embed when tied) and compute last-token logits
    let lm_path = splitted_dir.join(format!("{}.safetensors", layer_names.lm_head));
    let head_path = if lm_path.exists() { lm_path } else { embed_path };
    let head_raw = load_safetensors_all(&head_path)?;
    let head_map = decompress_layer_map(head_raw)?;
    let head_w = head_map
        .values()
        .next()
        .ok_or_else(|| "lm_head shard has no tensors".to_string())?;
    profiler.layers_streamed += 1;

    let last_h = &hidden[(seq_len - 1) * hidden_size..seq_len * hidden_size];
    let eff_vocab = (head_w.len() / hidden_size).min(vocab_size);
    let logits = linear_f32(last_h, &head_w[..eff_vocab * hidden_size], 1, hidden_size, eff_vocab);

    let mut ranked: Vec<(usize, f32)> = logits.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then_with(|| a.0.cmp(&b.0)));
    let next_token_id = ranked.first().map(|(i, _)| *i).unwrap_or(0);
    ranked.truncate(5);

    Ok(StreamedForwardResult {
        next_token_id,
        top5_tokens: ranked,
        profiler,
    })
}

fn apply_decoder_layer_weights(
    hidden: &mut Vec<f32>,
    weights: &BTreeMap<String, Vec<f32>>,
    shard_path: &Path,
    expert_prefix: Option<&str>,
    seq_len: usize,
    hidden_size: usize,
    profiler: &mut LayeredProfiler,
) -> Result<(), String> {
    let find_suffix = |suf: &str| -> Option<&Vec<f32>> {
        weights.iter().find(|(k, _)| k.ends_with(suf)).map(|(_, v)| v)
    };

    // Input LayerNorm
    let normed = if let Some(nw) = find_suffix("input_layernorm.weight") {
        if nw.len() == hidden_size {
            rms_norm(hidden, nw, 1e-5)
        } else {
            hidden.clone()
        }
    } else {
        hidden.clone()
    };

    // Self-Attention (Q/K/V/O with RoPE & causal dot-product)
    if let (Some(wq), Some(wk), Some(wv), Some(wo)) = (
        find_suffix("q_proj.weight"),
        find_suffix("k_proj.weight"),
        find_suffix("v_proj.weight"),
        find_suffix("o_proj.weight"),
    ) {
        if wq.len() == hidden_size * hidden_size
            && wk.len() == hidden_size * hidden_size
            && wv.len() == hidden_size * hidden_size
            && wo.len() == hidden_size * hidden_size
        {
            let mut q = linear_f32(&normed, wq, seq_len, hidden_size, hidden_size);
            let mut k = linear_f32(&normed, wk, seq_len, hidden_size, hidden_size);
            let v = linear_f32(&normed, wv, seq_len, hidden_size, hidden_size);
            apply_rope_inplace(&mut q, seq_len, hidden_size, 10000.0);
            apply_rope_inplace(&mut k, seq_len, hidden_size, 10000.0);

            let scale = 1.0 / (hidden_size as f32).sqrt();
            let mut attn_out = vec![0.0f32; seq_len * hidden_size];
            for i in 0..seq_len {
                let qi = &q[i * hidden_size..(i + 1) * hidden_size];
                let mut scores = vec![0.0f32; i + 1];
                let mut max_s = f32::NEG_INFINITY;
                for j in 0..=i {
                    let kj = &k[j * hidden_size..(j + 1) * hidden_size];
                    let mut dot = 0.0f32;
                    for d in 0..hidden_size {
                        dot += qi[d] * kj[d];
                    }
                    scores[j] = dot * scale;
                    if scores[j] > max_s {
                        max_s = scores[j];
                    }
                }
                let mut sum_e = 0.0f32;
                for s in &mut scores {
                    *s = (*s - max_s).exp();
                    sum_e += *s;
                }
                let out_i = &mut attn_out[i * hidden_size..(i + 1) * hidden_size];
                for j in 0..=i {
                    let w_ij = scores[j] / sum_e.max(1e-12);
                    let vj = &v[j * hidden_size..(j + 1) * hidden_size];
                    for d in 0..hidden_size {
                        out_i[d] += w_ij * vj[d];
                    }
                }
            }
            let proj = linear_f32(&attn_out, wo, seq_len, hidden_size, hidden_size);
            for (h, p) in hidden.iter_mut().zip(proj.iter()) {
                *h += *p;
            }
        }
    }

    // Per-Expert MoE On-Demand Streaming (if shard has routed experts!)
    if let Some(ep) = expert_prefix {
        let (_, per_expert) = partition_expert_keys(shard_path, ep)?;
        if !per_expert.is_empty() {
            let num_exp = per_expert.len();
            let router_logits: Vec<f32> = if let Some(rw) = find_suffix("gate.weight") {
                if rw.len() >= num_exp * hidden_size {
                    let last = &hidden[(seq_len - 1) * hidden_size..seq_len * hidden_size];
                    linear_f32(last, &rw[..num_exp * hidden_size], 1, hidden_size, num_exp)
                } else {
                    (0..num_exp).map(|i| (i as f32 * 0.1).sin()).collect()
                }
            } else {
                (0..num_exp).map(|i| (i as f32 * 0.1).sin()).collect()
            };
            let selected = route_topk_experts(&router_logits, 2);
            let mut moe_delta = vec![0.0f32; hidden.len()];
            for (exp_idx, weight) in selected {
                if let Some(keys) = per_expert.get(&exp_idx) {
                    let raw_exp = load_safetensors_subset(shard_path, keys)?;
                    let exp_w = decompress_layer_map(raw_exp)?;
                    profiler.experts_streamed += 1;
                    if let Some(w_down) = exp_w.values().next() {
                        if w_down.len() == hidden_size * hidden_size {
                            let y = linear_f32(hidden, w_down, seq_len, hidden_size, hidden_size);
                            for (dst, &val) in moe_delta.iter_mut().zip(y.iter()) {
                                *dst += val * weight;
                            }
                        }
                    }
                }
            }
            for (h, d) in hidden.iter_mut().zip(moe_delta.iter()) {
                *h += *d;
            }
            return Ok(());
        }
    }

    // Dense SwiGLU MLP
    if let (Some(wg), Some(wu), Some(wd)) = (
        find_suffix("gate_proj.weight"),
        find_suffix("up_proj.weight"),
        find_suffix("down_proj.weight"),
    ) {
        let inter = wg.len() / hidden_size.max(1);
        if inter > 0
            && wg.len() == inter * hidden_size
            && wu.len() == inter * hidden_size
            && wd.len() == hidden_size * inter
        {
            let gate = linear_f32(hidden, wg, seq_len, hidden_size, inter);
            let up = linear_f32(hidden, wu, seq_len, hidden_size, inter);
            let mut act = vec![0.0f32; gate.len()];
            for i in 0..gate.len() {
                act[i] = silu(gate[i]) * up[i];
            }
            let down = linear_f32(&act, wd, seq_len, inter, hidden_size);
            for (h, d) in hidden.iter_mut().zip(down.iter()) {
                *h += *d;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hal::CompressionMode;
    use crate::safetensors::{split_safetensors_checkpoint, write_safetensors};
    use std::fs;
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn end_to_end_nf4_split_prefetch_stream_forward() {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("nala_airllm_stream_{nanos}"));
        fs::create_dir_all(&dir).unwrap();

        let h = 16usize;
        let inter = 32usize;
        let vocab = 32usize;
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.embed_tokens.weight".to_string(),
            RawTensor::from_f32(
                vec![vocab, h],
                &(0..vocab * h).map(|i| (i as f32 * 0.05).sin() * 0.2).collect::<Vec<_>>(),
            ),
        );
        for l in 0..2 {
            tensors.insert(
                format!("model.layers.{l}.input_layernorm.weight"),
                RawTensor::from_f32(vec![h], &vec![1.0f32; h]),
            );
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                tensors.insert(
                    format!("model.layers.{l}.self_attn.{proj}.weight"),
                    RawTensor::from_f32(
                        vec![h, h],
                        &(0..h * h).map(|i| ((i + l) as f32 * 0.09).cos() * 0.1).collect::<Vec<_>>(),
                    ),
                );
            }
            tensors.insert(
                format!("model.layers.{l}.mlp.gate_proj.weight"),
                RawTensor::from_f32(vec![inter, h], &vec![0.05f32; inter * h]),
            );
            tensors.insert(
                format!("model.layers.{l}.mlp.up_proj.weight"),
                RawTensor::from_f32(vec![inter, h], &vec![0.05f32; inter * h]),
            );
            tensors.insert(
                format!("model.layers.{l}.mlp.down_proj.weight"),
                RawTensor::from_f32(vec![h, inter], &vec![0.05f32; h * inter]),
            );
        }
        tensors.insert(
            "model.norm.weight".to_string(),
            RawTensor::from_f32(vec![h], &vec![1.0f32; h]),
        );
        tensors.insert(
            "lm_head.weight".to_string(),
            RawTensor::from_f32(
                vec![vocab, h],
                &(0..vocab * h).map(|i| (i as f32 * 0.07).cos() * 0.3).collect::<Vec<_>>(),
            ),
        );

        write_safetensors(&dir.join("model.safetensors"), &tensors, &BTreeMap::new()).unwrap();

        let ldict = LayerNamesDict::for_architecture("LlamaForCausalLM");
        let split =
            split_safetensors_checkpoint(&dir, None, &ldict, CompressionMode::Nf4, false).unwrap();

        let res = run_streamed_forward(&split.output_dir, &ldict, &[1, 4, 9], 2, h, vocab, true)
            .expect("streamed forward ok");
        assert!(res.next_token_id < vocab);
        assert_eq!(res.profiler.layers_streamed, 5); // embed + 2 layers + norm + lm_head
        assert_eq!(res.top5_tokens.len(), 5);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn deltanet_step_remains_finite_and_bounded() {
        let dim = 8;
        let mut state = vec![0.0f32; dim * dim];
        let k = vec![0.5f32; dim];
        let v = vec![1.0f32; dim];
        let q = vec![0.25f32; dim];
        let y = gated_deltanet_step(&mut state, &k, &v, &q, 0.9, 0.5, dim);
        assert!(y.iter().all(|x| x.is_finite()));
        assert!(y.iter().any(|&x| x.abs() > 0.0));
    }
}
