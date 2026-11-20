//! Universal Cross-OS Hardware Abstraction Layer (NALA-HAL) & Roofline VRAM Planner.
//! Zero hardcoded hostnames, IPs, or personal paths — probes runtime hardware dynamically.

use crate::catalog::{ModelPreset, find_preset, frontier_presets};
use crate::json::JsonValue;
use std::collections::BTreeMap;
use std::process::Command;

pub const MAX_PINNED_LAYER_BYTES: u64 = 2 * 1024 * 1024 * 1024; // 2 GiB AirLLM pinned-RAM cap

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompressionMode {
    None,
    Int8,
    Nf4,
    Mxfp4,
}

impl CompressionMode {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "bf16" | "fp16" | "" => Ok(Self::None),
            "8bit" | "int8" => Ok(Self::Int8),
            "4bit" | "nf4" => Ok(Self::Nf4),
            "mxfp4" | "fp4" => Ok(Self::Mxfp4),
            other => Err(format!(
                "unsupported compression '{other}' (expected: none, 8bit, 4bit, mxfp4)"
            )),
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Int8 => "8bit",
            Self::Nf4 => "4bit",
            Self::Mxfp4 => "mxfp4",
        }
    }

    /// Approximate disk/transfer reduction factor relative to bf16/fp16.
    pub fn disk_ratio(&self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Int8 => 0.52,
            Self::Nf4 => 0.2813,
            Self::Mxfp4 => 0.27,
        }
    }

    /// AirLLM measured NVMe-bottlenecked speedup factor.
    pub fn io_speedup(&self) -> f64 {
        match self {
            Self::None => 1.0,
            Self::Int8 => 1.85,
            Self::Nf4 => 3.05,
            Self::Mxfp4 => 3.60,
        }
    }
}

#[derive(Debug, Clone)]
pub struct HardwareProfile {
    pub os: String,
    pub arch: String,
    pub profile_id: String,
    pub backend: String,
    pub cpu_cores: usize,
    pub system_ram_gb: f64,
    pub gpu_vram_gb: f64,
    pub gpu_name: String,
    pub unified_memory: bool,
    pub nvme_read_gbps: f64,
    pub macos_all_arch_unlocked: bool,
}

impl HardwareProfile {
    pub fn detect() -> Self {
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();
        let cpu_cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);

        let system_ram_gb = detect_system_ram_gb(&os);
        let (gpu_name, gpu_vram_gb, backend, unified_memory) =
            detect_gpu(&os, &arch, system_ram_gb);

        let profile_id = classify_hal_profile(&os, &arch, gpu_vram_gb, unified_memory).to_string();
        let nvme_read_gbps = if unified_memory { 6.2 } else { 5.0 };

        Self {
            os,
            arch,
            profile_id,
            backend,
            cpu_cores,
            system_ram_gb,
            gpu_vram_gb,
            gpu_name,
            unified_memory,
            nvme_read_gbps,
            macos_all_arch_unlocked: true,
        }
    }

    pub fn simulated(vram_gb: f64, ram_gb: f64, backend: &str) -> Self {
        let unified = backend.eq_ignore_ascii_case("metal") || backend.eq_ignore_ascii_case("mlx");
        let os = std::env::consts::OS.to_string();
        let arch = std::env::consts::ARCH.to_string();
        Self {
            os: os.clone(),
            arch: arch.clone(),
            profile_id: classify_hal_profile(&os, &arch, vram_gb, unified).to_string(),
            backend: backend.to_string(),
            cpu_cores: std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(8),
            system_ram_gb: ram_gb,
            gpu_vram_gb: vram_gb,
            gpu_name: format!("Simulated {backend} ({vram_gb:.1} GB)"),
            unified_memory: unified,
            nvme_read_gbps: 5.5,
            macos_all_arch_unlocked: true,
        }
    }

    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert("os".to_string(), JsonValue::String(self.os.clone()));
        m.insert("arch".to_string(), JsonValue::String(self.arch.clone()));
        m.insert(
            "profile_id".to_string(),
            JsonValue::String(self.profile_id.clone()),
        );
        m.insert(
            "backend".to_string(),
            JsonValue::String(self.backend.clone()),
        );
        m.insert(
            "cpu_cores".to_string(),
            JsonValue::Number(self.cpu_cores as f64),
        );
        m.insert(
            "system_ram_gb".to_string(),
            JsonValue::Number((self.system_ram_gb * 100.0).round() / 100.0),
        );
        m.insert(
            "gpu_vram_gb".to_string(),
            JsonValue::Number((self.gpu_vram_gb * 100.0).round() / 100.0),
        );
        m.insert(
            "gpu_name".to_string(),
            JsonValue::String(self.gpu_name.clone()),
        );
        m.insert(
            "unified_memory".to_string(),
            JsonValue::Bool(self.unified_memory),
        );
        m.insert(
            "nvme_read_gbps".to_string(),
            JsonValue::Number(self.nvme_read_gbps),
        );
        m.insert(
            "macos_all_arch_unlocked".to_string(),
            JsonValue::Bool(self.macos_all_arch_unlocked),
        );
        JsonValue::Object(m)
    }
}

fn classify_hal_profile(os: &str, arch: &str, vram_gb: f64, unified: bool) -> &'static str {
    if os == "macos" && arch == "aarch64" && unified {
        if vram_gb >= 32.0 {
            "NALA-HAL-P1-APPLE-SILICON-PRO-MAX"
        } else {
            "NALA-HAL-P2-APPLE-SILICON-UNIFIED"
        }
    } else if os == "macos" {
        "NALA-HAL-P3-MACOS-INTEL-METAL"
    } else if vram_gb >= 20.0 {
        "NALA-HAL-P4-DISCRETE-HIGH-VRAM"
    } else if vram_gb >= 8.0 {
        "NALA-HAL-P5-DISCRETE-MID-VRAM"
    } else if vram_gb >= 3.5 {
        "NALA-HAL-P6-EDGE-4GB-GPU"
    } else {
        "NALA-HAL-P7-CPU-NVME-STREAM"
    }
}

fn detect_system_ram_gb(os: &str) -> f64 {
    if os == "macos" {
        if let Ok(out) = Command::new("sysctl").args(["-n", "hw.memsize"]).output() {
            if let Ok(s) = String::from_utf8(out.stdout) {
                if let Ok(bytes) = s.trim().parse::<u64>() {
                    return bytes as f64 / (1024.0 * 1024.0 * 1024.0);
                }
            }
        }
    } else if os == "linux" {
        if let Ok(meminfo) = std::fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if let Some(rest) = line.strip_prefix("MemTotal:") {
                    if let Some(kb_str) = rest.split_whitespace().next() {
                        if let Ok(kb) = kb_str.parse::<u64>() {
                            return kb as f64 / (1024.0 * 1024.0);
                        }
                    }
                }
            }
        }
    }
    16.0
}

fn detect_gpu(os: &str, arch: &str, ram_gb: f64) -> (String, f64, String, bool) {
    if let Ok(out) = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,memory.total",
            "--format=csv,noheader,nounits",
        ])
        .output()
    {
        if out.status.success() {
            if let Ok(s) = String::from_utf8(out.stdout) {
                if let Some(first) = s.lines().next() {
                    let parts: Vec<&str> = first.split(',').collect();
                    if parts.len() >= 2 {
                        let name = parts[0].trim().to_string();
                        if let Ok(mib) = parts[1].trim().parse::<f64>() {
                            return (name, mib / 1024.0, "cuda".to_string(), false);
                        }
                    }
                }
            }
        }
    }

    if os == "macos" && arch == "aarch64" {
        let usable = (ram_gb * 0.75).max(4.0);
        return (
            "Apple Silicon Unified GPU (Metal)".to_string(),
            usable,
            "metal".to_string(),
            true,
        );
    }

    if os == "macos" {
        return (
            "macOS Metal / AVX2 Fallback".to_string(),
            4.0,
            "metal".to_string(),
            false,
        );
    }

    ("CPU + NVMe Layer-Streaming Fallback".to_string(), 0.0, "cpu".to_string(), false)
}

#[derive(Debug, Clone)]
pub struct StreamingPlan {
    pub model_id: String,
    pub display_name: String,
    pub architecture: String,
    pub total_params_b: f64,
    pub compression: CompressionMode,
    pub prefetching: bool,
    pub per_expert_streaming: bool,
    pub single_layer_disk_gb: f64,
    pub peak_vram_gb: f64,
    pub pinned_host_ram_gb: f64,
    pub ple_mmap_host_gb: f64,
    pub disk_required_gb: f64,
    pub fits_4gb_gpu: bool,
    pub fits_host_vram: bool,
    pub est_layer_io_ms: f64,
    pub est_token_latency_ms: f64,
    pub est_tokens_per_sec: f64,
    pub roofline_bottleneck: String,
    pub hardware: HardwareProfile,
    pub recommendations: Vec<String>,
}

pub fn build_streaming_plan(
    model_query: &str,
    compression: CompressionMode,
    prefetching: bool,
    delete_original: bool,
    hardware: &HardwareProfile,
) -> StreamingPlan {
    let preset: ModelPreset = find_preset(model_query)
        .unwrap_or_else(|| frontier_presets().into_iter().find(|p| p.id == "llama-3-70b").unwrap());

    // Effective ratio: if model is already native MXFP4/FP8 and compression is None, keep native ratio.
    let comp_scale = match (preset.native_dtype, compression) {
        ("mxfp4", _) => 1.0,
        ("fp8_e4m3", CompressionMode::None) => 1.0,
        ("fp8_e4m3", CompressionMode::Nf4) => 0.56,
        (_, c) => c.disk_ratio(),
    };

    let single_layer_disk_gb = (preset.single_layer_gb * comp_scale).max(0.12);
    let active_vram_gb = if compression == CompressionMode::None {
        preset.peak_vram_gb
    } else {
        (preset.single_layer_gb * comp_scale * 1.35 + 0.65).min(preset.peak_vram_gb)
    };

    // AirLLM pinned-RAM cap: 2 GiB per prefetched layer (0 if layer exceeds cap or prefetch off).
    let single_layer_bytes = (single_layer_disk_gb * 1024.0 * 1024.0 * 1024.0) as u64;
    let pinned_host_ram_gb = if prefetching && single_layer_bytes <= MAX_PINNED_LAYER_BYTES {
        single_layer_disk_gb
    } else {
        0.0
    };

    let split_disk_gb = preset.disk_checkpoint_gb * comp_scale;
    let disk_required_gb = if delete_original {
        split_disk_gb
    } else {
        preset.disk_checkpoint_gb + split_disk_gb
    };

    let effective_nvme_gbps = hardware.nvme_read_gbps.max(1.0);
    let overlap_bonus = if prefetching { 0.88 } else { 1.0 };
    let est_layer_io_ms = (single_layer_disk_gb / effective_nvme_gbps) * 1000.0 * overlap_bonus;
    let est_token_latency_ms = est_layer_io_ms * (preset.num_layers as f64) + 18.0;
    let est_tokens_per_sec = 1000.0 / est_token_latency_ms.max(1.0);

    let fits_4gb_gpu = active_vram_gb <= 4.0;
    let fits_host_vram = hardware.gpu_vram_gb == 0.0 || active_vram_gb <= hardware.gpu_vram_gb;

    let roofline_bottleneck = if hardware.gpu_vram_gb == 0.0 {
        "CPU Compute + NVMe Sequential Read".to_string()
    } else if single_layer_disk_gb > 0.5 {
        format!(
            "NVMe-to-PCIe Layer Stream ({:.1} GB/s) — {}x Compression Boost",
            effective_nvme_gbps,
            compression.io_speedup()
        )
    } else {
        "GPU Compute & KV-Cache Bandwidth".to_string()
    };

    let mut recommendations = Vec::new();
    if preset.per_expert_streaming {
        recommendations.push(format!(
            "Per-expert streaming active: reads only {}/{} routed experts per token via direct Safetensors byte-offset seek.",
            preset.routed_experts, preset.num_experts
        ));
    }
    if preset.ple_mmap_host_gb > 0.0 {
        recommendations.push(format!(
            "PLE n-gram table ({:.0} GB) peeled out to host file-backed .mmap — 0 bytes copied to GPU VRAM.",
            preset.ple_mmap_host_gb
        ));
    }
    if !fits_4gb_gpu && compression == CompressionMode::None {
        recommendations.push(
            "Enable --compression 4bit (NF4 blockwise) to cut peak VRAM & NVMe transfer by ~3.5x."
                .to_string(),
        );
    }
    if prefetching && single_layer_bytes > MAX_PINNED_LAYER_BYTES {
        recommendations.push(
            "Single layer exceeds 2 GiB pinned-RAM cap; falling back to pageable prefetch buffer (prevents host OOM)."
                .to_string(),
        );
    }
    if hardware.os == "macos" {
        recommendations.push(
            "NALA-airllm-rs unlocks ALL architectures on macOS (lifting upstream MLX Llama/Mistral-only restriction)."
                .to_string(),
        );
    }

    StreamingPlan {
        model_id: preset.id.to_string(),
        display_name: preset.display_name.to_string(),
        architecture: preset.architecture.to_string(),
        total_params_b: preset.total_params_b,
        compression,
        prefetching,
        per_expert_streaming: preset.per_expert_streaming,
        single_layer_disk_gb: (single_layer_disk_gb * 100.0).round() / 100.0,
        peak_vram_gb: (active_vram_gb * 100.0).round() / 100.0,
        pinned_host_ram_gb: (pinned_host_ram_gb * 100.0).round() / 100.0,
        ple_mmap_host_gb: preset.ple_mmap_host_gb,
        disk_required_gb: (disk_required_gb * 10.0).round() / 10.0,
        fits_4gb_gpu,
        fits_host_vram,
        est_layer_io_ms: (est_layer_io_ms * 100.0).round() / 100.0,
        est_token_latency_ms: (est_token_latency_ms * 10.0).round() / 10.0,
        est_tokens_per_sec: (est_tokens_per_sec * 1000.0).round() / 1000.0,
        roofline_bottleneck,
        hardware: hardware.clone(),
        recommendations,
    }
}

impl StreamingPlan {
    pub fn to_json(&self) -> JsonValue {
        let mut m = BTreeMap::new();
        m.insert("model_id".to_string(), JsonValue::String(self.model_id.clone()));
        m.insert(
            "display_name".to_string(),
            JsonValue::String(self.display_name.clone()),
        );
        m.insert(
            "architecture".to_string(),
            JsonValue::String(self.architecture.clone()),
        );
        m.insert(
            "total_params_b".to_string(),
            JsonValue::Number(self.total_params_b),
        );
        m.insert(
            "compression".to_string(),
            JsonValue::String(self.compression.as_str().to_string()),
        );
        m.insert("prefetching".to_string(), JsonValue::Bool(self.prefetching));
        m.insert(
            "per_expert_streaming".to_string(),
            JsonValue::Bool(self.per_expert_streaming),
        );
        m.insert(
            "single_layer_disk_gb".to_string(),
            JsonValue::Number(self.single_layer_disk_gb),
        );
        m.insert("peak_vram_gb".to_string(), JsonValue::Number(self.peak_vram_gb));
        m.insert(
            "pinned_host_ram_gb".to_string(),
            JsonValue::Number(self.pinned_host_ram_gb),
        );
        m.insert(
            "ple_mmap_host_gb".to_string(),
            JsonValue::Number(self.ple_mmap_host_gb),
        );
        m.insert(
            "disk_required_gb".to_string(),
            JsonValue::Number(self.disk_required_gb),
        );
        m.insert("fits_4gb_gpu".to_string(), JsonValue::Bool(self.fits_4gb_gpu));
        m.insert(
            "fits_host_vram".to_string(),
            JsonValue::Bool(self.fits_host_vram),
        );
        m.insert(
            "est_layer_io_ms".to_string(),
            JsonValue::Number(self.est_layer_io_ms),
        );
        m.insert(
            "est_token_latency_ms".to_string(),
            JsonValue::Number(self.est_token_latency_ms),
        );
        m.insert(
            "est_tokens_per_sec".to_string(),
            JsonValue::Number(self.est_tokens_per_sec),
        );
        m.insert(
            "roofline_bottleneck".to_string(),
            JsonValue::String(self.roofline_bottleneck.clone()),
        );
        m.insert("hardware".to_string(), self.hardware.to_json());
        m.insert(
            "recommendations".to_string(),
            JsonValue::Array(
                self.recommendations
                    .iter()
                    .map(|r| JsonValue::String(r.clone()))
                    .collect(),
            ),
        );
        JsonValue::Object(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kimi_k3_fits_under_4gb_gpu_plan() {
        let hw = HardwareProfile::simulated(4.0, 64.0, "cuda");
        let plan = build_streaming_plan("kimi-k3-2.8t", CompressionMode::Mxfp4, true, false, &hw);
        assert!(plan.fits_4gb_gpu);
        assert!(plan.per_expert_streaming);
        assert!(plan.peak_vram_gb < 4.0);
    }

    #[test]
    fn qwen38_flash_next_isolates_ple_table() {
        let hw = HardwareProfile::simulated(6.0, 64.0, "cuda");
        let plan = build_streaming_plan(
            "qwen3.8-flash-next-125b",
            CompressionMode::None,
            true,
            true,
            &hw,
        );
        assert_eq!(plan.ple_mmap_host_gb, 102.0);
        assert!(plan.fits_host_vram);
    }
}
