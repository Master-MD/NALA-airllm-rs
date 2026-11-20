//! Pure Safe Rust Streamed Layer-Wise LoRA Trainer, Packed-Expert LoRA,
//! Fused Chunked Linear Cross-Entropy (never materializes [N, vocab] logits),
//! AdamW Optimizer & SFT Dataset Loader (.jsonl / .json / .txt with -100 prefix masking).

use crate::json::{JsonValue, parse_json};
use crate::safetensors::{RawTensor, load_safetensors_all, write_safetensors};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;

pub const DEFAULT_LORA_TARGETS: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

pub const FLASH_NEXT_LORA_TARGETS: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "shared_expert_gate",
    "input_mix_weight_down",
    "input_mix_weight_up",
    "block_inject_weight",
    "key_proj",
    "value_proj",
];

/// Fused Chunked Linear Cross-Entropy forward + exact analytical backward (`d_hidden`).
/// Exact parity with AirLLM `chunked_linear_cross_entropy`: walks `vocab` in slices of
/// `chunk_size` using online `logaddexp` so `[N, 248320]` logits are NEVER allocated!
///
/// - `hidden`: `[n_tokens, hidden_dim]`
/// - `weight`: `[vocab_size, hidden_dim]` (lives on CPU / streamed in chunks)
/// - `labels`: `[n_tokens]` (`-100` / `ignore_index` masked out)
pub fn chunked_linear_cross_entropy_fwd_bwd(
    hidden: &[f32],
    weight: &[f32],
    labels: &[i64],
    n_tokens: usize,
    hidden_dim: usize,
    vocab_size: usize,
    chunk_size: usize,
    ignore_index: i64,
) -> Result<(f32, Vec<f32>), String> {
    if hidden.len() != n_tokens * hidden_dim {
        return Err("hidden shape mismatch".to_string());
    }
    if weight.len() != vocab_size * hidden_dim {
        return Err("weight shape mismatch".to_string());
    }
    if labels.len() != n_tokens {
        return Err("labels length mismatch".to_string());
    }
    let cs = if chunk_size == 0 { vocab_size } else { chunk_size };

    let mut logz = vec![f32::NEG_INFINITY; n_tokens];
    let mut target_logit = vec![0.0f32; n_tokens];
    let mut n_valid = 0usize;
    for &lbl in labels {
        if lbl != ignore_index {
            n_valid += 1;
        }
    }

    // Pass 1: online logsumexp + target logit across vocab chunks
    let mut start = 0;
    while start < vocab_size {
        let end = (start + cs).min(vocab_size);
        let c_len = end - start;
        for t in 0..n_tokens {
            if labels[t] == ignore_index {
                continue;
            }
            let h_row = &hidden[t * hidden_dim..(t + 1) * hidden_dim];
            let mut max_l = f32::NEG_INFINITY;
            let mut chunk_logits = vec![0.0f32; c_len];
            for v in 0..c_len {
                let w_row = &weight[(start + v) * hidden_dim..(start + v + 1) * hidden_dim];
                let mut dot = 0.0f32;
                for d in 0..hidden_dim {
                    dot += h_row[d] * w_row[d];
                }
                chunk_logits[v] = dot;
                if dot > max_l {
                    max_l = dot;
                }
                if labels[t] == (start + v) as i64 {
                    target_logit[t] = dot;
                }
            }
            let mut sum_exp = 0.0f32;
            for &l in &chunk_logits {
                sum_exp += (l - max_l).exp();
            }
            let chunk_lse = max_l + sum_exp.ln();
            logz[t] = logaddexp(logz[t], chunk_lse);
        }
        start = end;
    }

    let mut loss_sum = 0.0f32;
    for t in 0..n_tokens {
        if labels[t] != ignore_index {
            loss_sum += logz[t] - target_logit[t];
        }
    }
    let denom = n_valid.max(1) as f32;
    let loss = loss_sum / denom;

    // Pass 2: recompute chunks in backward to accumulate exact d_hidden without storing logits
    let mut d_hidden = vec![0.0f32; n_tokens * hidden_dim];
    let inv_denom = 1.0 / denom;
    start = 0;
    while start < vocab_size {
        let end = (start + cs).min(vocab_size);
        let c_len = end - start;
        for t in 0..n_tokens {
            if labels[t] == ignore_index {
                continue;
            }
            let h_row = &hidden[t * hidden_dim..(t + 1) * hidden_dim];
            let dh_row = &mut d_hidden[t * hidden_dim..(t + 1) * hidden_dim];
            for v in 0..c_len {
                let w_row = &weight[(start + v) * hidden_dim..(start + v + 1) * hidden_dim];
                let mut dot = 0.0f32;
                for d in 0..hidden_dim {
                    dot += h_row[d] * w_row[d];
                }
                let mut prob = (dot - logz[t]).exp();
                if labels[t] == (start + v) as i64 {
                    prob -= 1.0;
                }
                let scaled = prob * inv_denom;
                for d in 0..hidden_dim {
                    dh_row[d] += scaled * w_row[d];
                }
            }
        }
        start = end;
    }

    Ok((loss, d_hidden))
}

#[inline]
fn logaddexp(a: f32, b: f32) -> f32 {
    if a == f32::NEG_INFINITY {
        return b;
    }
    if b == f32::NEG_INFINITY {
        return a;
    }
    let m = a.max(b);
    m + ((a - m).exp() + (b - m).exp()).ln()
}

/// Deterministic XorShift64 PRNG for Kaiming-uniform LoRA initialization.
struct Rng64(u64);
impl Rng64 {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }
    fn uniform_neg1_1(&mut self) -> f32 {
        let u = (self.next_u64() >> 40) as f32 / ((1u64 << 24) as f32);
        u * 2.0 - 1.0
    }
}

/// Minimal LoRA Linear adapter (`y = x W^T + (x A^T) B^T * (alpha / r)`).
/// Frozen base weight `W` streams from disk; `lora_A` and `lora_B` + AdamW moments stay resident.
#[derive(Debug, Clone)]
pub struct LoRALinearAdapter {
    pub name: String,
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
    pub alpha: f32,
    pub lora_a: Vec<f32>, // [r, in_features]
    pub lora_b: Vec<f32>, // [out_features, r]
    pub grad_a: Vec<f32>,
    pub grad_b: Vec<f32>,
    pub m_a: Vec<f32>,
    pub v_a: Vec<f32>,
    pub m_b: Vec<f32>,
    pub v_b: Vec<f32>,
    pub step: u64,
}

impl LoRALinearAdapter {
    pub fn new(name: &str, in_features: usize, out_features: usize, rank: usize, alpha: f32, seed: u64) -> Self {
        let r = rank.max(1);
        let mut rng = Rng64(seed.max(1));
        let bound = (1.0 / in_features.max(1) as f32).sqrt();
        let lora_a: Vec<f32> = (0..r * in_features)
            .map(|_| rng.uniform_neg1_1() * bound)
            .collect();
        let lora_b = vec![0.0f32; out_features * r];
        Self {
            name: name.to_string(),
            in_features,
            out_features,
            rank: r,
            alpha,
            grad_a: vec![0.0; lora_a.len()],
            grad_b: vec![0.0; lora_b.len()],
            m_a: vec![0.0; lora_a.len()],
            v_a: vec![0.0; lora_a.len()],
            m_b: vec![0.0; lora_b.len()],
            v_b: vec![0.0; lora_b.len()],
            lora_a,
            lora_b,
            step: 0,
        }
    }

    pub fn scaling(&self) -> f32 {
        self.alpha / (self.rank as f32)
    }

    /// Forward: `y = x W^T + (x A^T) B^T * scaling`.
    /// `x`: `[n, in_features]`, `base_w`: `[out_features, in_features]`.
    pub fn forward(&self, x: &[f32], base_w: &[f32], n_tokens: usize) -> Vec<f32> {
        let mut y = vec![0.0f32; n_tokens * self.out_features];
        let mut xa = vec![0.0f32; n_tokens * self.rank];
        let scale = self.scaling();

        for t in 0..n_tokens {
            let x_row = &x[t * self.in_features..(t + 1) * self.in_features];
            // Base frozen linear
            for o in 0..self.out_features {
                let w_row = &base_w[o * self.in_features..(o + 1) * self.in_features];
                let mut acc = 0.0f32;
                for i in 0..self.in_features {
                    acc += x_row[i] * w_row[i];
                }
                y[t * self.out_features + o] = acc;
            }
            // x A^T -> [rank]
            for r in 0..self.rank {
                let a_row = &self.lora_a[r * self.in_features..(r + 1) * self.in_features];
                let mut acc = 0.0f32;
                for i in 0..self.in_features {
                    acc += x_row[i] * a_row[i];
                }
                xa[t * self.rank + r] = acc;
            }
            // + (x A^T) B^T * scale
            for o in 0..self.out_features {
                let b_row = &self.lora_b[o * self.rank..(o + 1) * self.rank];
                let mut acc = 0.0f32;
                for r in 0..self.rank {
                    acc += xa[t * self.rank + r] * b_row[r];
                }
                y[t * self.out_features + o] += acc * scale;
            }
        }
        y
    }

    /// Analytical backward: accumulates `grad_a`, `grad_b`, and returns `d_x` (`[n, in_features]`).
    pub fn backward(
        &mut self,
        x: &[f32],
        base_w: &[f32],
        grad_y: &[f32],
        n_tokens: usize,
    ) -> Vec<f32> {
        let scale = self.scaling();
        let mut d_x = vec![0.0f32; n_tokens * self.in_features];

        for t in 0..n_tokens {
            let x_row = &x[t * self.in_features..(t + 1) * self.in_features];
            let gy_row = &grad_y[t * self.out_features..(t + 1) * self.out_features];
            let dx_row = &mut d_x[t * self.in_features..(t + 1) * self.in_features];

            // Compute z = x A^T [rank]
            let mut z = vec![0.0f32; self.rank];
            for r in 0..self.rank {
                let a_row = &self.lora_a[r * self.in_features..(r + 1) * self.in_features];
                let mut acc = 0.0f32;
                for i in 0..self.in_features {
                    acc += x_row[i] * a_row[i];
                }
                z[r] = acc;
            }

            // grad_B[o, r] += scale * gy[o] * z[r]
            // dz[r] = scale * sum_o gy[o] * B[o, r]
            let mut dz = vec![0.0f32; self.rank];
            for o in 0..self.out_features {
                let gy_o = gy_row[o];
                let w_row = &base_w[o * self.in_features..(o + 1) * self.in_features];
                for i in 0..self.in_features {
                    dx_row[i] += gy_o * w_row[i];
                }
                for r in 0..self.rank {
                    self.grad_b[o * self.rank + r] += scale * gy_o * z[r];
                    dz[r] += scale * gy_o * self.lora_b[o * self.rank + r];
                }
            }

            // grad_A[r, i] += dz[r] * x[i] and dx[i] += sum_r dz[r] * A[r, i]
            for r in 0..self.rank {
                let dz_r = dz[r];
                for i in 0..self.in_features {
                    self.grad_a[r * self.in_features + i] += dz_r * x_row[i];
                    dx_row[i] += dz_r * self.lora_a[r * self.in_features + i];
                }
            }
        }
        d_x
    }

    pub fn zero_grad(&mut self) {
        self.grad_a.fill(0.0);
        self.grad_b.fill(0.0);
    }

    /// AdamW step (beta1=0.9, beta2=0.999, eps=1e-8).
    pub fn adamw_step(&mut self, lr: f32, weight_decay: f32) {
        self.step += 1;
        let t = self.step as i32;
        let b1 = 0.9f32;
        let b2 = 0.999f32;
        let eps = 1e-8f32;
        let bc1 = 1.0 - b1.powi(t);
        let bc2 = 1.0 - b2.powi(t);

        let update = |params: &mut [f32], grads: &[f32], m: &mut [f32], v: &mut [f32]| {
            for i in 0..params.len() {
                let g = grads[i];
                m[i] = b1 * m[i] + (1.0 - b1) * g;
                v[i] = b2 * v[i] + (1.0 - b2) * g * g;
                let m_hat = m[i] / bc1;
                let v_hat = v[i] / bc2;
                params[i] -= lr * (m_hat / (v_hat.sqrt() + eps) + weight_decay * params[i]);
            }
        };

        update(&mut self.lora_a, &self.grad_a, &mut self.m_a, &mut self.v_a);
        update(&mut self.lora_b, &self.grad_b, &mut self.m_b, &mut self.v_b);
    }

    pub fn save_safetensors(&self, path: &Path) -> Result<u64, String> {
        let mut tensors = BTreeMap::new();
        tensors.insert(
            format!("{}.lora_A", self.name),
            RawTensor::from_f32(vec![self.rank, self.in_features], &self.lora_a),
        );
        tensors.insert(
            format!("{}.lora_B", self.name),
            RawTensor::from_f32(vec![self.out_features, self.rank], &self.lora_b),
        );
        let mut meta = BTreeMap::new();
        meta.insert("adapter".to_string(), "nala-airllm-rs-lora".to_string());
        meta.insert("rank".to_string(), self.rank.to_string());
        meta.insert("alpha".to_string(), self.alpha.to_string());
        write_safetensors(path, &tensors, &meta)
    }

    pub fn load_safetensors(&mut self, path: &Path) -> Result<(), String> {
        let loaded = load_safetensors_all(path)?;
        let ka = format!("{}.lora_A", self.name);
        let kb = format!("{}.lora_B", self.name);
        let a = loaded
            .get(&ka)
            .ok_or_else(|| format!("missing {ka}"))?
            .to_f32_vec()?;
        let b = loaded
            .get(&kb)
            .ok_or_else(|| format!("missing {kb}"))?
            .to_f32_vec()?;
        if a.len() != self.lora_a.len() || b.len() != self.lora_b.len() {
            return Err("LoRA checkpoint dimension mismatch".to_string());
        }
        self.lora_a = a;
        self.lora_b = b;
        Ok(())
    }
}

/// Exact SFT training record (parity with `airllm/lora_data.py`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SftRecord {
    pub text: String,
    pub mask_prefix: Option<String>,
}

pub fn record_from_json(val: &JsonValue) -> Result<SftRecord, String> {
    if let Some(s) = val.as_str() {
        let trimmed = s.trim();
        if trimmed.is_empty() {
            return Err("empty string example".to_string());
        }
        return Ok(SftRecord {
            text: trimmed.to_string(),
            mask_prefix: None,
        });
    }
    let obj = val
        .as_object()
        .ok_or_else(|| "expected a JSON object or string".to_string())?;

    if let Some(t) = obj.get("text").and_then(|v| v.as_str()) {
        if !t.is_empty() {
            return Ok(SftRecord {
                text: t.to_string(),
                mask_prefix: None,
            });
        }
    }

    let prompt = obj
        .get("prompt")
        .or_else(|| obj.get("instruction"))
        .and_then(|v| v.as_str())
        .unwrap_or("");
    let extra = obj.get("input").and_then(|v| v.as_str()).unwrap_or("");
    let completion = obj
        .get("completion")
        .or_else(|| obj.get("output"))
        .or_else(|| obj.get("response"))
        .and_then(|v| v.as_str())
        .unwrap_or("");

    if !prompt.is_empty() || !completion.is_empty() {
        let mut prefix_parts = Vec::new();
        if !prompt.trim().is_empty() {
            prefix_parts.push(prompt.trim());
        }
        if !extra.trim().is_empty() {
            prefix_parts.push(extra.trim());
        }
        let prefix = prefix_parts.join("\n");
        let text = if !prefix.is_empty() && !completion.is_empty() {
            let sep = if prefix.ends_with('\n') || prefix.ends_with(' ') {
                ""
            } else {
                "\n"
            };
            format!("{prefix}{sep}{completion}")
        } else if !prefix.is_empty() {
            prefix.clone()
        } else {
            completion.to_string()
        };
        if text.trim().is_empty() {
            return Err("prompt/completion example is empty".to_string());
        }
        return Ok(SftRecord {
            text,
            mask_prefix: if prefix.is_empty() { None } else { Some(prefix) },
        });
    }

    if let Some(msgs) = obj.get("messages").and_then(|v| v.as_array()) {
        let mut lines = Vec::new();
        let mut last_assistant: Option<String> = None;
        for m in msgs {
            let role = m
                .get("role")
                .and_then(|v| v.as_str())
                .unwrap_or("user")
                .trim();
            let content = m
                .get("content")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .trim();
            if content.is_empty() {
                continue;
            }
            lines.push(format!("{role}: {content}"));
            if role == "assistant" {
                last_assistant = Some(content.to_string());
            }
        }
        if lines.is_empty() {
            return Err("messages example is empty".to_string());
        }
        let text = lines.join("\n");
        let mask_prefix = last_assistant.and_then(|a| {
            if text.ends_with(&a) {
                Some(text[..text.len() - a.len()].to_string())
            } else {
                None
            }
        });
        return Ok(SftRecord { text, mask_prefix });
    }

    Err("each example needs `text`, `prompt`/`completion`, `instruction`/`output`, or `messages`".to_string())
}

/// Load SFT records from `.jsonl`, `.json`, or `.txt` (exact parity with `airllm/lora_data.py`).
pub fn load_sft_records(path: &Path) -> Result<Vec<SftRecord>, String> {
    let raw = fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    let raw = raw.strip_prefix('\u{feff}').unwrap_or(&raw);
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();

    let mut records = Vec::new();
    if ext == "txt" {
        for block in raw.split("\n\n") {
            let b = block.trim();
            if !b.is_empty() {
                records.push(SftRecord {
                    text: b.to_string(),
                    mask_prefix: None,
                });
            }
        }
    } else if ext == "json" {
        let parsed = parse_json(raw).map_err(|e| format!("{}: {e}", path.display()))?;
        let items: Vec<JsonValue> = match &parsed {
            JsonValue::Array(a) => a.clone(),
            JsonValue::Object(m) => {
                if let Some(JsonValue::Array(a)) =
                    m.get("data").or_else(|| m.get("examples")).or_else(|| m.get("rows"))
                {
                    a.clone()
                } else {
                    vec![parsed]
                }
            }
            _ => return Err("JSON file must be an object, list, or {\"data\":[...]}".to_string()),
        };
        for (i, item) in items.iter().enumerate() {
            records.push(
                record_from_json(item).map_err(|e| format!("{}:{i}: {e}", path.display()))?,
            );
        }
    } else {
        for (idx, line) in raw.lines().enumerate() {
            let trimmed = line.trim();
            if trimmed.is_empty() || trimmed.starts_with('#') {
                continue;
            }
            let parsed = parse_json(trimmed)
                .map_err(|e| format!("{}:{}: {e}", path.display(), idx + 1))?;
            records.push(
                record_from_json(&parsed)
                    .map_err(|e| format!("{}:{}: {e}", path.display(), idx + 1))?,
            );
        }
    }
    if records.is_empty() {
        return Err(format!("no examples in {}", path.display()));
    }
    Ok(records)
}

#[derive(Debug, Clone)]
pub struct LoRaBudgetEstimate {
    pub model_id: String,
    pub rank: usize,
    pub seq_len: usize,
    pub ce_chunk_size: usize,
    pub trainable_params_m: f64,
    pub adapter_vram_mb: f64,
    pub adamw_vram_mb: f64,
    pub streamed_layer_vram_gb: f64,
    pub chunked_ce_vram_mb: f64,
    pub total_peak_vram_gb: f64,
    pub fits_6gb_vram: bool,
}

pub fn estimate_streamed_lora_budget(
    model_id: &str,
    num_layers: usize,
    hidden_size: usize,
    single_layer_gb: f64,
    rank: usize,
    seq_len: usize,
    ce_chunk_size: usize,
    flash_next_targets: bool,
) -> LoRaBudgetEstimate {
    let target_count = if flash_next_targets {
        FLASH_NEXT_LORA_TARGETS.len()
    } else {
        DEFAULT_LORA_TARGETS.len()
    };
    let params_per_linear = 2 * rank * hidden_size;
    let total_trainable = (num_layers * target_count * params_per_linear) as f64;
    let trainable_params_m = total_trainable / 1e6;
    let adapter_vram_mb = (total_trainable * 2.0) / (1024.0 * 1024.0); // bf16
    let adamw_vram_mb = (total_trainable * 8.0) / (1024.0 * 1024.0); // fp32 m + v
    let chunked_ce_vram_mb =
        ((seq_len * ce_chunk_size * 4) as f64) / (1024.0 * 1024.0); // fp32 logits slice
    let total_peak_vram_gb =
        single_layer_gb + (adapter_vram_mb + adamw_vram_mb + chunked_ce_vram_mb) / 1024.0 + 0.35;

    LoRaBudgetEstimate {
        model_id: model_id.to_string(),
        rank,
        seq_len,
        ce_chunk_size,
        trainable_params_m: (trainable_params_m * 100.0).round() / 100.0,
        adapter_vram_mb: (adapter_vram_mb * 10.0).round() / 10.0,
        adamw_vram_mb: (adamw_vram_mb * 10.0).round() / 10.0,
        streamed_layer_vram_gb: (single_layer_gb * 100.0).round() / 100.0,
        chunked_ce_vram_mb: (chunked_ce_vram_mb * 10.0).round() / 10.0,
        total_peak_vram_gb: (total_peak_vram_gb * 100.0).round() / 100.0,
        fits_6gb_vram: total_peak_vram_gb <= 6.0,
    }
}

impl LoRaBudgetEstimate {
    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert("model_id".to_string(), JsonValue::String(self.model_id.clone()));
        m.insert("rank".to_string(), JsonValue::Number(self.rank as f64));
        m.insert("seq_len".to_string(), JsonValue::Number(self.seq_len as f64));
        m.insert(
            "ce_chunk_size".to_string(),
            JsonValue::Number(self.ce_chunk_size as f64),
        );
        m.insert(
            "trainable_params_m".to_string(),
            JsonValue::Number(self.trainable_params_m),
        );
        m.insert(
            "adapter_vram_mb".to_string(),
            JsonValue::Number(self.adapter_vram_mb),
        );
        m.insert(
            "adamw_vram_mb".to_string(),
            JsonValue::Number(self.adamw_vram_mb),
        );
        m.insert(
            "streamed_layer_vram_gb".to_string(),
            JsonValue::Number(self.streamed_layer_vram_gb),
        );
        m.insert(
            "chunked_ce_vram_mb".to_string(),
            JsonValue::Number(self.chunked_ce_vram_mb),
        );
        m.insert(
            "total_peak_vram_gb".to_string(),
            JsonValue::Number(self.total_peak_vram_gb),
        );
        m.insert(
            "fits_6gb_vram".to_string(),
            JsonValue::Bool(self.fits_6gb_vram),
        );
        JsonValue::Object(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn chunked_ce_matches_finite_difference_gradient() {
        let n = 3;
        let h = 4;
        let v = 10;
        let hidden: Vec<f32> = (0..n * h).map(|i| (i as f32 * 0.17).sin() * 0.5).collect();
        let weight: Vec<f32> = (0..v * h).map(|i| (i as f32 * 0.23).cos() * 0.4).collect();
        let labels = vec![2i64, -100, 7];

        // Compare chunk_size=3 vs full vocab chunk_size=10
        let (loss_chunked, dh_chunked) =
            chunked_linear_cross_entropy_fwd_bwd(&hidden, &weight, &labels, n, h, v, 3, -100)
                .unwrap();
        let (loss_full, dh_full) =
            chunked_linear_cross_entropy_fwd_bwd(&hidden, &weight, &labels, n, h, v, 10, -100)
                .unwrap();

        assert!((loss_chunked - loss_full).abs() < 1e-5);
        for (a, b) in dh_chunked.iter().zip(dh_full.iter()) {
            assert!((a - b).abs() < 1e-5);
        }

        // Numerical gradient check
        let eps = 1e-3f32;
        for idx in 0..hidden.len() {
            let mut hp = hidden.clone();
            hp[idx] += eps;
            let (lp, _) =
                chunked_linear_cross_entropy_fwd_bwd(&hp, &weight, &labels, n, h, v, 3, -100)
                    .unwrap();
            let mut hm = hidden.clone();
            hm[idx] -= eps;
            let (lm, _) =
                chunked_linear_cross_entropy_fwd_bwd(&hm, &weight, &labels, n, h, v, 3, -100)
                    .unwrap();
            let num_g = (lp - lm) / (2.0 * eps);
            assert!(
                (num_g - dh_chunked[idx]).abs() < 1e-3,
                "grad mismatch at {idx}: num={num_g} vs analytic={}",
                dh_chunked[idx]
            );
        }
    }

    #[test]
    fn lora_training_reduces_loss_with_adamw() {
        let n = 2;
        let in_f = 6;
        let out_f = 6;
        let base_w: Vec<f32> = (0..out_f * in_f)
            .map(|i| (i as f32 * 0.11).sin() * 0.3)
            .collect();
        let x: Vec<f32> = (0..n * in_f).map(|i| (i as f32 * 0.29).cos()).collect();
        let target: Vec<f32> = (0..n * out_f).map(|i| (i as f32 * 0.07).sin()).collect();

        let mut lora = LoRALinearAdapter::new("layers.0.q_proj", in_f, out_f, 4, 8.0, 42);
        let mut first_loss = 0.0;
        let mut last_loss = 0.0;

        for step in 0..25 {
            lora.zero_grad();
            let y = lora.forward(&x, &base_w, n);
            let mut loss = 0.0f32;
            let mut gy = vec![0.0f32; y.len()];
            for i in 0..y.len() {
                let diff = y[i] - target[i];
                loss += 0.5 * diff * diff;
                gy[i] = diff;
            }
            if step == 0 {
                first_loss = loss;
            }
            last_loss = loss;
            let _dx = lora.backward(&x, &base_w, &gy, n);
            lora.adamw_step(0.05, 0.0);
        }
        assert!(
            last_loss < first_loss * 0.3,
            "LoRA did not converge: start={first_loss} end={last_loss}"
        );
    }

    #[test]
    fn parses_all_sft_record_formats() {
        let j1 = parse_json(r#"{"prompt":"Question?","completion":"Answer."}"#).unwrap();
        let r1 = record_from_json(&j1).unwrap();
        assert_eq!(r1.text, "Question?\nAnswer.");
        assert_eq!(r1.mask_prefix.as_deref(), Some("Question?"));

        let j2 = parse_json(
            r#"{"messages":[{"role":"user","content":"Hi"},{"role":"assistant","content":"Hello"}]}"#,
        )
        .unwrap();
        let r2 = record_from_json(&j2).unwrap();
        assert_eq!(r2.text, "user: Hi\nassistant: Hello");
        assert_eq!(r2.mask_prefix.as_deref(), Some("user: Hi\nassistant: "));
    }
}
