//! Pure Safe Rust Binary Safetensors Engine, Layer Splitter, Random-Access Per-Expert Reader,
//! Zero-Copy Hardlink Passthrough, and Contiguous PLE N-Gram File-Backed Table Persister.

use crate::catalog::LayerNamesDict;
use crate::hal::CompressionMode;
use crate::json::{JsonValue, parse_json};
use crate::quant::{
    bf16_to_f32, f16_to_f32, f32_to_bf16, fp8_e4m3_to_f32, quantize_int8_blockwise, quantize_nf4,
    should_load_verbatim,
};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

pub const MAX_HEADER_BYTES: usize = 64 * 1024 * 1024; // 64 MiB safety ceiling

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: (u64, u64),
}

impl TensorMeta {
    pub fn byte_len(&self) -> u64 {
        self.data_offsets.1.saturating_sub(self.data_offsets.0)
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().copied().product::<usize>()
    }
}

#[derive(Debug, Clone)]
pub struct SafetensorsHeader {
    pub header_len: u64,
    pub data_start: u64,
    pub tensors: BTreeMap<String, TensorMeta>,
    pub metadata: BTreeMap<String, String>,
}

/// Read ONLY the Safetensors JSON header without reading any tensor payloads.
pub fn read_safetensors_header(path: &Path) -> Result<SafetensorsHeader, String> {
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let file_len = file
        .metadata()
        .map_err(|e| format!("stat {}: {e}", path.display()))?
        .len();
    if file_len < 8 {
        return Err(format!(
            "invalid safetensors file {} (< 8 bytes)",
            path.display()
        ));
    }
    let mut len_buf = [0u8; 8];
    file.read_exact(&mut len_buf)
        .map_err(|e| format!("read header len: {e}"))?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len == 0 || header_len as usize > MAX_HEADER_BYTES || 8 + header_len > file_len {
        return Err(format!(
            "invalid safetensors header_len {header_len} for file size {file_len}"
        ));
    }
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes)
        .map_err(|e| format!("read header json: {e}"))?;
    let header_str =
        std::str::from_utf8(&header_bytes).map_err(|e| format!("header utf8 error: {e}"))?;
    let parsed = parse_json(header_str).map_err(|e| format!("header json error: {e}"))?;
    let obj = parsed
        .as_object()
        .ok_or_else(|| "safetensors header must be a JSON object".to_string())?;

    let data_start = 8 + header_len;
    let payload_len = file_len - data_start;
    let mut tensors = BTreeMap::new();
    let mut metadata = BTreeMap::new();

    for (k, v) in obj {
        if k == "__metadata__" {
            if let Some(meta_obj) = v.as_object() {
                for (mk, mv) in meta_obj {
                    if let Some(s) = mv.as_str() {
                        metadata.insert(mk.clone(), s.to_string());
                    }
                }
            }
            continue;
        }
        let dtype = v
            .get("dtype")
            .and_then(|d| d.as_str())
            .ok_or_else(|| format!("tensor {k} missing dtype"))?
            .to_string();
        let shape: Vec<usize> = v
            .get("shape")
            .and_then(|s| s.as_array())
            .ok_or_else(|| format!("tensor {k} missing shape"))?
            .iter()
            .filter_map(|x| x.as_u64().map(|n| n as usize))
            .collect();
        let offs = v
            .get("data_offsets")
            .and_then(|o| o.as_array())
            .ok_or_else(|| format!("tensor {k} missing data_offsets"))?;
        if offs.len() != 2 {
            return Err(format!("tensor {k} data_offsets must have 2 elements"));
        }
        let start = offs[0]
            .as_u64()
            .ok_or_else(|| format!("tensor {k} invalid start offset"))?;
        let end = offs[1]
            .as_u64()
            .ok_or_else(|| format!("tensor {k} invalid end offset"))?;
        if end < start || end > payload_len {
            return Err(format!(
                "tensor {k} offsets [{start}, {end}] exceed payload {payload_len}"
            ));
        }
        tensors.insert(
            k.clone(),
            TensorMeta {
                name: k.clone(),
                dtype,
                shape,
                data_offsets: (start, end),
            },
        );
    }

    Ok(SafetensorsHeader {
        header_len,
        data_start,
        tensors,
        metadata,
    })
}

#[derive(Debug, Clone)]
pub struct RawTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data: Vec<u8>,
}

impl RawTensor {
    pub fn from_f32(shape: Vec<usize>, vals: &[f32]) -> Self {
        let mut data = Vec::with_capacity(vals.len() * 4);
        for &v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        Self {
            dtype: "F32".to_string(),
            shape,
            data,
        }
    }

    pub fn from_bf16_from_f32(shape: Vec<usize>, vals: &[f32]) -> Self {
        let mut data = Vec::with_capacity(vals.len() * 2);
        for &v in vals {
            data.extend_from_slice(&f32_to_bf16(v).to_le_bytes());
        }
        Self {
            dtype: "BF16".to_string(),
            shape,
            data,
        }
    }

    pub fn to_f32_vec(&self) -> Result<Vec<f32>, String> {
        match self.dtype.as_str() {
            "F32" => Ok(self
                .data
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect()),
            "BF16" => Ok(self
                .data
                .chunks_exact(2)
                .map(|c| bf16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            "F16" => Ok(self
                .data
                .chunks_exact(2)
                .map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]])))
                .collect()),
            "F8_E4M3" => Ok(self.data.iter().map(|&b| fp8_e4m3_to_f32(b)).collect()),
            other => Err(format!("cannot decode dtype '{other}' directly to f32")),
        }
    }
}

/// Write a standard `.safetensors` file atomically (write `.tmp` then rename).
pub fn write_safetensors(
    path: &Path,
    tensors: &BTreeMap<String, RawTensor>,
    metadata: &BTreeMap<String, String>,
) -> Result<u64, String> {
    let mut header_obj = BTreeMap::new();
    if !metadata.is_empty() {
        let mut meta_map = BTreeMap::new();
        for (k, v) in metadata {
            meta_map.insert(k.clone(), JsonValue::String(v.clone()));
        }
        header_obj.insert("__metadata__".to_string(), JsonValue::Object(meta_map));
    }

    let mut offset: u64 = 0;
    for (name, t) in tensors {
        let end = offset + t.data.len() as u64;
        let mut entry = BTreeMap::new();
        entry.insert("dtype".to_string(), JsonValue::String(t.dtype.clone()));
        entry.insert(
            "shape".to_string(),
            JsonValue::Array(t.shape.iter().map(|&d| JsonValue::Number(d as f64)).collect()),
        );
        entry.insert(
            "data_offsets".to_string(),
            JsonValue::Array(vec![
                JsonValue::Number(offset as f64),
                JsonValue::Number(end as f64),
            ]),
        );
        header_obj.insert(name.clone(), JsonValue::Object(entry));
        offset = end;
    }

    let header_json = JsonValue::Object(header_obj).to_json_string();
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("create_dir_all {}: {e}", parent.display()))?;
    }
    let tmp = path.with_extension("safetensors.tmp");
    {
        let mut file = File::create(&tmp).map_err(|e| format!("create {}: {e}", tmp.display()))?;
        file.write_all(&header_len.to_le_bytes())
            .map_err(|e| format!("write header len: {e}"))?;
        file.write_all(header_bytes)
            .map_err(|e| format!("write header: {e}"))?;
        for t in tensors.values() {
            file.write_all(&t.data)
                .map_err(|e| format!("write tensor payload: {e}"))?;
        }
        file.sync_all().map_err(|e| format!("sync {}: {e}", tmp.display()))?;
    }
    fs::rename(&tmp, path).map_err(|e| format!("rename {} -> {}: {e}", tmp.display(), path.display()))?;
    Ok(8 + header_len + offset)
}

/// Random-access subset reader (`load_layer_subset` parity):
/// Seeks straight to the requested keys' byte offsets (critical for Kimi K3 per-expert streaming!).
pub fn load_safetensors_subset(
    path: &Path,
    keys: &[String],
) -> Result<BTreeMap<String, RawTensor>, String> {
    let header = read_safetensors_header(path)?;
    let mut file = File::open(path).map_err(|e| format!("open {}: {e}", path.display()))?;
    let mut out = BTreeMap::new();

    for key in keys {
        let meta = header
            .tensors
            .get(key)
            .ok_or_else(|| format!("tensor '{key}' not found in {}", path.display()))?;
        let byte_len = meta.byte_len() as usize;
        file.seek(SeekFrom::Start(header.data_start + meta.data_offsets.0))
            .map_err(|e| format!("seek '{key}': {e}"))?;
        let mut buf = vec![0u8; byte_len];
        file.read_exact(&mut buf)
            .map_err(|e| format!("read '{key}': {e}"))?;
        out.insert(
            key.clone(),
            RawTensor {
                dtype: meta.dtype.clone(),
                shape: meta.shape.clone(),
                data: buf,
            },
        );
    }
    Ok(out)
}

pub fn load_safetensors_all(path: &Path) -> Result<BTreeMap<String, RawTensor>, String> {
    let header = read_safetensors_header(path)?;
    let keys: Vec<String> = header.tensors.keys().cloned().collect();
    load_safetensors_subset(path, &keys)
}

/// Exact AirLLM longest-prefix owner resolution: ensures nested `cpu_resident` tables
/// (e.g., `model.language_model.layers.3.ple.ple_embedding.ngram_embedding.`) are NOT
/// swallowed by their parent decoder layer (`model.language_model.layers.3.`).
pub fn layer_owner<'a>(key: &str, layer_prefixes: &'a [String]) -> Option<&'a str> {
    let mut best: Option<&'a str> = None;
    for prefix in layer_prefixes {
        if key.starts_with(prefix.as_str()) {
            if best.map_or(true, |b| prefix.len() > b.len()) {
                best = Some(prefix.as_str());
            }
        }
    }
    best
}

/// Extract contiguous `*.ngram_embedding.shard_N.weight` shard index if matched.
pub fn parse_ngram_shard_key(key: &str) -> Option<(&str, usize)> {
    let marker = ".ngram_embedding.shard_";
    let pos = key.find(marker)?;
    let prefix = &key[..pos + ".ngram_embedding".len()];
    let rest = &key[pos + marker.len()..];
    let idx_str = rest.strip_suffix(".weight")?;
    let idx: usize = idx_str.parse().ok()?;
    Some((prefix, idx))
}

/// Extract shard ordinal from Hugging Face shard names like `model-00003-of-00064.safetensors`.
fn shard_ordinal(filename: &str) -> i64 {
    let parts: Vec<&str> = filename.split('-').collect();
    if parts.len() > 1 {
        if let Ok(n) = parts[1].parse::<i64>() {
            return n;
        }
    }
    -1
}

/// Discover `cpu_resident` + `cpu_resident_marker` prefixes from checkpoint index keys.
pub fn cpu_resident_module_names(
    layer_names: &LayerNamesDict,
    index_keys: &[String],
) -> Vec<String> {
    let mut names = layer_names.cpu_resident.clone();
    if let Some(marker) = &layer_names.cpu_resident_marker {
        for key in index_keys {
            if let Some(pos) = key.find(marker.as_str()) {
                let prefix = key[..pos + marker.len()].to_string();
                if !names.contains(&prefix) {
                    names.push(prefix);
                }
            }
        }
    }
    names
}

/// Compress a layer's state dict into NF4 (`4bit`) or INT8 (`8bit`) safetensors representation.
pub fn compress_layer_tensors(
    tensors: BTreeMap<String, RawTensor>,
    compression: CompressionMode,
) -> Result<BTreeMap<String, RawTensor>, String> {
    if matches!(compression, CompressionMode::None | CompressionMode::Mxfp4) {
        return Ok(tensors);
    }
    let mut out = BTreeMap::new();
    for (k, t) in tensors {
        if should_load_verbatim(&k, &t.dtype) || t.shape.iter().copied().product::<usize>() < 64 {
            out.insert(k, t);
            continue;
        }
        let Ok(vals) = t.to_f32_vec() else {
            out.insert(k, t);
            continue;
        };
        match compression {
            CompressionMode::Nf4 => {
                let q = quantize_nf4(&vals, 64);
                let mut abs_bytes = Vec::with_capacity(q.absmax.len() * 4);
                for &a in &q.absmax {
                    abs_bytes.extend_from_slice(&a.to_le_bytes());
                }
                let mut shape_bytes = Vec::with_capacity(t.shape.len() * 8);
                for &d in &t.shape {
                    shape_bytes.extend_from_slice(&(d as u64).to_le_bytes());
                }
                out.insert(
                    k.clone(),
                    RawTensor {
                        dtype: "U8".to_string(),
                        shape: vec![q.packed.len()],
                        data: q.packed,
                    },
                );
                out.insert(
                    format!("{k}.4bit.absmax"),
                    RawTensor {
                        dtype: "F32".to_string(),
                        shape: vec![q.absmax.len()],
                        data: abs_bytes,
                    },
                );
                out.insert(
                    format!("{k}.4bit.shape"),
                    RawTensor {
                        dtype: "U64".to_string(),
                        shape: vec![t.shape.len()],
                        data: shape_bytes,
                    },
                );
            }
            CompressionMode::Int8 => {
                let q = quantize_int8_blockwise(&vals, 2048);
                let packed_u8: Vec<u8> = q.packed.iter().map(|&v| v as u8).collect();
                let mut abs_bytes = Vec::with_capacity(q.absmax.len() * 4);
                for &a in &q.absmax {
                    abs_bytes.extend_from_slice(&a.to_le_bytes());
                }
                let mut shape_bytes = Vec::with_capacity(t.shape.len() * 8);
                for &d in &t.shape {
                    shape_bytes.extend_from_slice(&(d as u64).to_le_bytes());
                }
                out.insert(
                    k.clone(),
                    RawTensor {
                        dtype: "I8".to_string(),
                        shape: vec![packed_u8.len()],
                        data: packed_u8,
                    },
                );
                out.insert(
                    format!("{k}.8bit.absmax"),
                    RawTensor {
                        dtype: "F32".to_string(),
                        shape: vec![q.absmax.len()],
                        data: abs_bytes,
                    },
                );
                out.insert(
                    format!("{k}.8bit.shape"),
                    RawTensor {
                        dtype: "U64".to_string(),
                        shape: vec![t.shape.len()],
                        data: shape_bytes,
                    },
                );
            }
            _ => {
                out.insert(k, t);
            }
        }
    }
    Ok(out)
}

#[derive(Debug, Clone)]
pub struct SplitReport {
    pub output_dir: PathBuf,
    pub total_layers_written: usize,
    pub passthrough_hardlinked: usize,
    pub ngram_tables_persisted: usize,
    pub total_bytes_written: u64,
    pub layer_files: Vec<String>,
}

/// Split a single-file or multi-shard Safetensors checkpoint into per-layer AirLLM shards
/// with zero-copy hardlink passthrough, PLE n-gram raw `.mmap` peeling, and optional NF4/INT8 compression.
pub fn split_safetensors_checkpoint(
    checkpoint_dir: &Path,
    output_base: Option<&Path>,
    layer_names: &LayerNamesDict,
    compression: CompressionMode,
    delete_original: bool,
) -> Result<SplitReport, String> {
    let dir_name = match compression {
        CompressionMode::None | CompressionMode::Mxfp4 => "splitted_model".to_string(),
        CompressionMode::Int8 => "splitted_model.8bit".to_string(),
        CompressionMode::Nf4 => "splitted_model.4bit".to_string(),
    };
    let saving_path = output_base
        .unwrap_or(checkpoint_dir)
        .join(&dir_name);
    fs::create_dir_all(&saving_path)
        .map_err(|e| format!("mkdir {}: {e}", saving_path.display()))?;

    // Build weight_map (tensor_name -> shard_filename)
    let mut weight_map: BTreeMap<String, String> = BTreeMap::new();
    let index_file = checkpoint_dir.join("model.safetensors.index.json");
    let single_file = checkpoint_dir.join("model.safetensors");
    if index_file.exists() {
        let raw = fs::read_to_string(&index_file)
            .map_err(|e| format!("read {}: {e}", index_file.display()))?;
        let parsed = parse_json(&raw).map_err(|e| format!("parse index json: {e}"))?;
        let wm = parsed
            .get("weight_map")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "index.json missing weight_map".to_string())?;
        for (k, v) in wm {
            if let Some(fname) = v.as_str() {
                weight_map.insert(k.clone(), fname.to_string());
            }
        }
    } else if single_file.exists() {
        let header = read_safetensors_header(&single_file)?;
        for k in header.tensors.keys() {
            weight_map.insert(k.clone(), "model.safetensors".to_string());
        }
    } else {
        return Err(format!(
            "No model.safetensors or model.safetensors.index.json found in {}",
            checkpoint_dir.display()
        ));
    }

    // Filter dropped prefixes (e.g. `mtp.*` on Qwen3.5 / Qwen4-Exp)
    if !layer_names.drop_prefixes.is_empty() {
        weight_map.retain(|k, _| {
            !layer_names
                .drop_prefixes
                .iter()
                .any(|dp| k.starts_with(dp.as_str()))
        });
    }

    let index_keys: Vec<String> = weight_map.keys().cloned().collect();

    // Determine decoder layer count dynamically
    let layer_prefix_dot = format!("{}.", layer_names.layer_prefix);
    let mut layer_indices = BTreeSet::new();
    for k in &index_keys {
        if let Some(rest) = k.strip_prefix(&layer_prefix_dot) {
            if let Some(num_str) = rest.split('.').next() {
                if let Ok(idx) = num_str.parse::<usize>() {
                    layer_indices.insert(idx);
                }
            }
        }
    }
    let n_layers = layer_indices.len();

    let mut raw_layers: Vec<String> = Vec::new();
    if let Some(rot) = &layer_names.rotary_pos_emb {
        raw_layers.push(rot.clone());
    }
    raw_layers.push(layer_names.embed.clone());
    for i in 0..n_layers {
        raw_layers.push(format!("{}.{i}", layer_names.layer_prefix));
    }
    raw_layers.push(layer_names.norm.clone());
    raw_layers.push(layer_names.lm_head.clone());
    for r in &layer_names.resident {
        raw_layers.push(r.clone());
    }
    for c in cpu_resident_module_names(layer_names, &index_keys) {
        if !raw_layers.contains(&c) {
            raw_layers.push(c);
        }
    }

    let mut layers: Vec<String> = raw_layers.into_iter().map(|l| format!("{l}.")).collect();
    // Retain only layers that actually own at least one tensor (handles tied embeddings cleanly)
    let owned: BTreeSet<String> = index_keys
        .iter()
        .filter_map(|k| layer_owner(k, &layers).map(|s| s.to_string()))
        .collect();
    layers.retain(|l| owned.contains(l));

    // Sort by (last_shard_ordinal, -prefix_len) so nested PLE tables extract first
    layers.sort_by(|a, b| {
        let last_a = weight_map
            .iter()
            .filter(|(k, _)| k.starts_with(a.as_str()))
            .map(|(_, v)| shard_ordinal(v))
            .max()
            .unwrap_or(-1);
        let last_b = weight_map
            .iter()
            .filter(|(k, _)| k.starts_with(b.as_str()))
            .map(|(_, v)| shard_ordinal(v))
            .max()
            .unwrap_or(-1);
        last_a.cmp(&last_b).then_with(|| b.len().cmp(&a.len()))
    });

    // Detect zero-copy hardlink passthrough shards (exact one-module-per-shard, no compression)
    let mut shard_contents: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (k, v) in &weight_map {
        shard_contents.entry(v.clone()).or_default().push(k.clone());
    }
    let mut passthrough: BTreeMap<String, String> = BTreeMap::new();
    if matches!(compression, CompressionMode::None | CompressionMode::Mxfp4) {
        for layer in &layers {
            let files: BTreeSet<&String> = weight_map
                .iter()
                .filter(|(k, _)| layer_owner(k, &layers) == Some(layer.as_str()))
                .map(|(_, v)| v)
                .collect();
            if files.len() == 1 {
                let only_file = (*files.iter().next().unwrap()).clone();
                if shard_contents
                    .get(&only_file)
                    .map_or(false, |ks| ks.iter().all(|k| k.starts_with(layer.as_str())))
                {
                    passthrough.insert(layer.clone(), only_file);
                }
            }
        }
    }

    let mut total_layers_written = 0usize;
    let mut passthrough_hardlinked = 0usize;
    let mut ngram_tables_persisted = 0usize;
    let mut total_bytes_written = 0u64;
    let mut layer_files = Vec::new();
    let mut original_files_used = BTreeSet::new();

    for layer in &layers {
        // Case 1: PLE n-gram table -> stream contiguous shards into flat file-backed `.mmap`
        if let Some(marker) = &layer_names.cpu_resident_marker {
            if layer.contains(marker.as_str()) {
                let bytes = persist_ngram_table_from_shards(
                    layer,
                    &weight_map,
                    checkpoint_dir,
                    &saving_path,
                )?;
                if bytes > 0 {
                    ngram_tables_persisted += 1;
                    total_bytes_written += bytes;
                    layer_files.push(format!("{layer}mmap"));
                    continue;
                }
            }
        }

        // Case 2: Zero-copy hardlink passthrough
        let out_file = saving_path.join(format!("{layer}safetensors"));
        let done_marker = saving_path.join(format!("{layer}safetensors.done"));
        if let Some(src_name) = passthrough.get(layer) {
            let src_path = checkpoint_dir.join(src_name);
            if !done_marker.exists() {
                if out_file.exists() {
                    let _ = fs::remove_file(&out_file);
                }
                if fs::hard_link(&src_path, &out_file).is_err() {
                    fs::copy(&src_path, &out_file)
                        .map_err(|e| format!("copy passthrough {}: {e}", src_path.display()))?;
                }
                fs::write(&done_marker, b"ok").map_err(|e| format!("write done marker: {e}"))?;
            }
            passthrough_hardlinked += 1;
            total_layers_written += 1;
            layer_files.push(format!("{layer}safetensors"));
            continue;
        }

        // Case 3: Collect all tensors owned by `layer` across referenced shards
        let mut by_file: BTreeMap<String, Vec<String>> = BTreeMap::new();
        for (k, fname) in &weight_map {
            if layer_owner(k, &layers) == Some(layer.as_str()) {
                by_file.entry(fname.clone()).or_default().push(k.clone());
                original_files_used.insert(fname.clone());
            }
        }
        let mut layer_tensors = BTreeMap::new();
        for (fname, keys) in by_file {
            let src_path = checkpoint_dir.join(&fname);
            let loaded = load_safetensors_subset(&src_path, &keys)?;
            layer_tensors.extend(loaded);
        }

        let compressed = compress_layer_tensors(layer_tensors, compression)?;
        let mut meta = BTreeMap::new();
        meta.insert("format".to_string(), "nala-airllm-rs".to_string());
        meta.insert("compression".to_string(), compression.as_str().to_string());
        let written = write_safetensors(&out_file, &compressed, &meta)?;
        fs::write(&done_marker, b"ok").map_err(|e| format!("write done marker: {e}"))?;

        total_bytes_written += written;
        total_layers_written += 1;
        layer_files.push(format!("{layer}safetensors"));
    }

    if delete_original {
        for fname in original_files_used {
            let p = checkpoint_dir.join(&fname);
            if p.exists() {
                let _ = fs::remove_file(&p);
            }
        }
    }

    Ok(SplitReport {
        output_dir: saving_path,
        total_layers_written,
        passthrough_hardlinked,
        ngram_tables_persisted,
        total_bytes_written,
        layer_files,
    })
}

/// Stream contiguous `*.ngram_embedding.shard_N.weight` shards into a flat `<layer>mmap` file
/// without holding two copies in RAM (exact parity with `persist_ngram_mmap`).
pub fn persist_ngram_table_from_shards(
    layer_prefix: &str,
    weight_map: &BTreeMap<String, String>,
    checkpoint_dir: &Path,
    saving_path: &Path,
) -> Result<u64, String> {
    let mut shards: Vec<(usize, String, String)> = Vec::new();
    for (k, fname) in weight_map {
        if !k.starts_with(layer_prefix) {
            continue;
        }
        if let Some((_, idx)) = parse_ngram_shard_key(k) {
            shards.push((idx, k.clone(), fname.clone()));
        }
    }
    if shards.is_empty() {
        return Ok(0);
    }
    shards.sort_by_key(|(idx, _, _)| *idx);
    for (expected, (got, _, _)) in shards.iter().enumerate() {
        if *got != expected {
            return Err(format!(
                "ngram embedding shards for {layer_prefix} are not contiguous 0..{}: missing {expected}",
                shards.len() - 1
            ));
        }
    }

    let raw_path = saving_path.join(format!("{layer_prefix}mmap"));
    let tmp_path = saving_path.join(format!("{layer_prefix}mmap.tmp"));
    let meta_path = saving_path.join(format!("{layer_prefix}mmap.json"));
    let done_path = saving_path.join(format!("{layer_prefix}mmap.done"));

    let mut total_rows = 0usize;
    let mut cols = 0usize;
    let mut dtype_tag = String::new();
    let mut total_bytes = 0u64;

    {
        let mut out = File::create(&tmp_path)
            .map_err(|e| format!("create {}: {e}", tmp_path.display()))?;
        for (_, key, fname) in &shards {
            let src_path = checkpoint_dir.join(fname);
            let loaded = load_safetensors_subset(&src_path, std::slice::from_ref(key))?;
            let tensor = loaded
                .get(key)
                .ok_or_else(|| format!("missing shard tensor {key}"))?;
            if tensor.shape.len() != 2 {
                return Err(format!("ngram shard {key} must be 2D"));
            }
            if cols == 0 {
                cols = tensor.shape[1];
                dtype_tag = tensor.dtype.clone();
            } else if tensor.shape[1] != cols {
                return Err(format!(
                    "ngram shard {key} column mismatch {} != {cols}",
                    tensor.shape[1]
                ));
            }
            total_rows += tensor.shape[0];
            total_bytes += tensor.data.len() as u64;
            out.write_all(&tensor.data)
                .map_err(|e| format!("write ngram mmap: {e}"))?;
        }
        out.sync_all().map_err(|e| format!("sync ngram mmap: {e}"))?;
    }
    fs::rename(&tmp_path, &raw_path)
        .map_err(|e| format!("rename ngram mmap: {e}"))?;

    let mut meta_obj = BTreeMap::new();
    meta_obj.insert("rows".to_string(), JsonValue::Number(total_rows as f64));
    meta_obj.insert("cols".to_string(), JsonValue::Number(cols as f64));
    meta_obj.insert("dtype".to_string(), JsonValue::String(dtype_tag));
    meta_obj.insert("nbytes".to_string(), JsonValue::Number(total_bytes as f64));
    fs::write(&meta_path, JsonValue::Object(meta_obj).to_json_pretty())
        .map_err(|e| format!("write mmap.json: {e}"))?;
    fs::write(&done_path, b"ok").map_err(|e| format!("write mmap.done: {e}"))?;

    Ok(total_bytes)
}

/// Direct file-seek gather of specific row indices from a persisted `.mmap` n-gram table
/// (0 bytes copied for unused rows; works with 102 GB tables on 16 GB RAM hosts!).
pub fn gather_ngram_rows(
    saving_path: &Path,
    layer_name: &str,
    row_indices: &[usize],
) -> Result<(Vec<f32>, usize), String> {
    let meta_path = saving_path.join(format!("{layer_name}.mmap.json"));
    let raw_path = saving_path.join(format!("{layer_name}.mmap"));
    let raw_meta = fs::read_to_string(&meta_path)
        .map_err(|e| format!("read {}: {e}", meta_path.display()))?;
    let parsed = parse_json(&raw_meta).map_err(|e| format!("parse mmap.json: {e}"))?;
    let rows = parsed.get("rows").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let cols = parsed.get("cols").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let dtype = parsed
        .get("dtype")
        .and_then(|v| v.as_str())
        .unwrap_or("F32");
    let elem_size: usize = match dtype {
        "F32" | "fp32" => 4,
        "BF16" | "bf16" | "F16" | "fp16" => 2,
        other => return Err(format!("unsupported mmap dtype '{other}'")),
    };
    let row_bytes = cols * elem_size;
    let mut file = File::open(&raw_path).map_err(|e| format!("open {}: {e}", raw_path.display()))?;
    let mut out = Vec::with_capacity(row_indices.len() * cols);
    let mut row_buf = vec![0u8; row_bytes];

    for &idx in row_indices {
        if idx >= rows {
            return Err(format!("ngram row index {idx} out of bounds (rows={rows})"));
        }
        file.seek(SeekFrom::Start((idx * row_bytes) as u64))
            .map_err(|e| format!("seek ngram row {idx}: {e}"))?;
        file.read_exact(&mut row_buf)
            .map_err(|e| format!("read ngram row {idx}: {e}"))?;
        match dtype {
            "F32" | "fp32" => {
                for c in row_buf.chunks_exact(4) {
                    out.push(f32::from_le_bytes([c[0], c[1], c[2], c[3]]));
                }
            }
            "BF16" | "bf16" => {
                for c in row_buf.chunks_exact(2) {
                    out.push(bf16_to_f32(u16::from_le_bytes([c[0], c[1]])));
                }
            }
            "F16" | "fp16" => {
                for c in row_buf.chunks_exact(2) {
                    out.push(f16_to_f32(u16::from_le_bytes([c[0], c[1]])));
                }
            }
            _ => {}
        }
    }
    Ok((out, cols))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_test_dir(tag: &str) -> PathBuf {
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!("nala_airllm_rs_{tag}_{nanos}"));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn longest_prefix_peels_ple_out_of_decoder_layer() {
        let layers = vec![
            "model.language_model.layers.0.".to_string(),
            "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.".to_string(),
        ];
        assert_eq!(
            layer_owner(
                "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight",
                &layers
            ),
            Some("model.language_model.layers.0.ple.ple_embedding.ngram_embedding.")
        );
        assert_eq!(
            layer_owner("model.language_model.layers.0.self_attn.q_proj.weight", &layers),
            Some("model.language_model.layers.0.")
        );
    }

    #[test]
    fn split_checkpoint_with_ple_mmap_and_random_access_seek() {
        let dir = temp_test_dir("split_ple");
        let mut tensors = BTreeMap::new();
        tensors.insert(
            "model.language_model.embed_tokens.weight".to_string(),
            RawTensor::from_f32(vec![16, 8], &vec![0.25f32; 128]),
        );
        tensors.insert(
            "model.language_model.layers.0.self_attn.q_proj.weight".to_string(),
            RawTensor::from_f32(vec![8, 8], &vec![0.5f32; 64]),
        );
        tensors.insert(
            "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_0.weight"
                .to_string(),
            RawTensor::from_f32(vec![4, 8], &(0..32).map(|i| i as f32).collect::<Vec<_>>()),
        );
        tensors.insert(
            "model.language_model.layers.0.ple.ple_embedding.ngram_embedding.shard_1.weight"
                .to_string(),
            RawTensor::from_f32(vec![4, 8], &(32..64).map(|i| i as f32).collect::<Vec<_>>()),
        );
        tensors.insert(
            "model.language_model.hyper_connection_mixer.weight".to_string(),
            RawTensor::from_f32(vec![8], &vec![1.0f32; 8]),
        );
        tensors.insert(
            "lm_head.weight".to_string(),
            RawTensor::from_f32(vec![16, 8], &vec![0.1f32; 128]),
        );
        tensors.insert(
            "mtp.head.weight".to_string(),
            RawTensor::from_f32(vec![4, 4], &vec![9.0f32; 16]),
        );

        write_safetensors(&dir.join("model.safetensors"), &tensors, &BTreeMap::new()).unwrap();

        let ldict = LayerNamesDict::for_architecture("Qwen4ExpForConditionalGeneration");
        let report =
            split_safetensors_checkpoint(&dir, None, &ldict, CompressionMode::None, false).unwrap();

        assert_eq!(report.ngram_tables_persisted, 1);
        assert!(report.total_layers_written >= 4);

        // Verify mtp.* was dropped and PLE table was peeled out of layer 0
        let l0_hdr = read_safetensors_header(
            &report
                .output_dir
                .join("model.language_model.layers.0.safetensors"),
        )
        .unwrap();
        assert!(l0_hdr.tensors.contains_key("model.language_model.layers.0.self_attn.q_proj.weight"));
        assert!(!l0_hdr.tensors.keys().any(|k| k.contains("ngram_embedding")));

        // Verify direct file-seek row gather from merged 8x8 PLE mmap
        let (rows, cols) = gather_ngram_rows(
            &report.output_dir,
            "model.language_model.layers.0.ple.ple_embedding.ngram_embedding",
            &[0, 5],
        )
        .unwrap();
        assert_eq!(cols, 8);
        assert_eq!(rows[0], 0.0);
        assert_eq!(rows[8], 40.0); // row 5 * 8 cols = 40.0

        let _ = fs::remove_dir_all(&dir);
    }
}
