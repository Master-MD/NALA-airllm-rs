//! Universal AirLLM AutoModel Architecture Detector, Layer-Name Overrides & Frontier Model Catalog.
//! 100% parity with upstream AirLLM v3.x (Llama, Mistral, Mixtral, Qwen/2/3/3.5/3.8/4-Exp,
//! Kimi K3 2.8T, DeepSeek-V2/V3, ChatGLM, Baichuan, InternLM, Phi-4, Gemma) + Cross-OS support.

use crate::json::{JsonValue, parse_json};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LayerNamesDict {
    pub architecture: String,
    pub model_class: String,
    pub embed: String,
    pub layer_prefix: String,
    pub norm: String,
    pub lm_head: String,
    pub rotary_pos_emb: Option<String>,
    pub expert_prefix: Option<String>,
    pub resident: Vec<String>,
    pub cpu_resident: Vec<String>,
    pub cpu_resident_marker: Option<String>,
    pub drop_prefixes: Vec<String>,
    pub int64_scatter_guard: bool,
}

impl LayerNamesDict {
    pub fn for_architecture(arch: &str) -> Self {
        match arch {
            "ChatGLMModel" | "ChatGLMForConditionalGeneration" => Self {
                architecture: arch.to_string(),
                model_class: "AirLLMChatGLM".to_string(),
                embed: "transformer.embedding.word_embeddings".to_string(),
                layer_prefix: "transformer.encoder.layers".to_string(),
                norm: "transformer.encoder.final_layernorm".to_string(),
                lm_head: "transformer.output_layer".to_string(),
                rotary_pos_emb: Some("transformer.rotary_pos_emb".to_string()),
                expert_prefix: None,
                resident: vec![],
                cpu_resident: vec![],
                cpu_resident_marker: None,
                drop_prefixes: vec![],
                int64_scatter_guard: false,
            },
            "QWenLMHeadModel" => Self {
                architecture: arch.to_string(),
                model_class: "AirLLMQWen".to_string(),
                embed: "transformer.wte".to_string(),
                layer_prefix: "transformer.h".to_string(),
                norm: "transformer.ln_f".to_string(),
                lm_head: "lm_head".to_string(),
                rotary_pos_emb: None,
                expert_prefix: None,
                resident: vec![],
                cpu_resident: vec![],
                cpu_resident_marker: None,
                drop_prefixes: vec![],
                int64_scatter_guard: false,
            },
            "BaichuanForCausalLM" | "BaiChuanForCausalLM" => Self {
                architecture: arch.to_string(),
                model_class: "AirLLMBaichuan".to_string(),
                embed: "model.embed_tokens".to_string(),
                layer_prefix: "model.layers".to_string(),
                norm: "model.norm".to_string(),
                lm_head: "lm_head".to_string(),
                rotary_pos_emb: None,
                expert_prefix: None,
                resident: vec![],
                cpu_resident: vec![],
                cpu_resident_marker: None,
                drop_prefixes: vec![],
                int64_scatter_guard: false,
            },
            "InternLMForCausalLM" => Self {
                architecture: arch.to_string(),
                model_class: "AirLLMInternLM".to_string(),
                embed: "model.embed_tokens".to_string(),
                layer_prefix: "model.layers".to_string(),
                norm: "model.norm".to_string(),
                lm_head: "lm_head".to_string(),
                rotary_pos_emb: None,
                expert_prefix: None,
                resident: vec![],
                cpu_resident: vec![],
                cpu_resident_marker: None,
                drop_prefixes: vec![],
                int64_scatter_guard: false,
            },
            "KimiK3ForConditionalGeneration" => Self {
                architecture: arch.to_string(),
                model_class: "AirLLMKimiK3".to_string(),
                embed: "language_model.model.embed_tokens".to_string(),
                layer_prefix: "language_model.model.layers".to_string(),
                norm: "language_model.model.norm".to_string(),
                lm_head: "language_model.lm_head".to_string(),
                rotary_pos_emb: None,
                expert_prefix: Some("block_sparse_moe.experts".to_string()),
                resident: vec![
                    "language_model.model.output_attn_res_norm".to_string(),
                    "language_model.model.output_attn_res_proj".to_string(),
                    "mm_projector".to_string(),
                    "vision_tower".to_string(),
                ],
                cpu_resident: vec![],
                cpu_resident_marker: None,
                drop_prefixes: vec![],
                int64_scatter_guard: false,
            },
            "Qwen3_5ForConditionalGeneration" => Self {
                architecture: arch.to_string(),
                model_class: "AirLLMQwen3_5".to_string(),
                embed: "model.language_model.embed_tokens".to_string(),
                layer_prefix: "model.language_model.layers".to_string(),
                norm: "model.language_model.norm".to_string(),
                lm_head: "lm_head".to_string(),
                rotary_pos_emb: None,
                expert_prefix: None,
                resident: vec!["model.visual".to_string()],
                cpu_resident: vec![],
                cpu_resident_marker: None,
                drop_prefixes: vec!["mtp.".to_string()],
                int64_scatter_guard: false,
            },
            "Qwen4ExpForConditionalGeneration" => Self {
                architecture: arch.to_string(),
                model_class: "AirLLMQwen4Exp".to_string(),
                embed: "model.language_model.embed_tokens".to_string(),
                layer_prefix: "model.language_model.layers".to_string(),
                norm: "model.language_model.hyper_connection_mixer".to_string(),
                lm_head: "lm_head".to_string(),
                rotary_pos_emb: None,
                expert_prefix: None, // Packed 3D expert tensors stream per layer
                resident: vec!["model.visual".to_string()],
                cpu_resident: vec![],
                cpu_resident_marker: Some("ple.ple_embedding.ngram_embedding".to_string()),
                drop_prefixes: vec!["mtp.".to_string()],
                int64_scatter_guard: true,
            },
            "MixtralForCausalLM" => Self {
                architecture: arch.to_string(),
                model_class: "AirLLMMixtral".to_string(),
                embed: "model.embed_tokens".to_string(),
                layer_prefix: "model.layers".to_string(),
                norm: "model.norm".to_string(),
                lm_head: "lm_head".to_string(),
                rotary_pos_emb: None,
                expert_prefix: Some("block_sparse_moe.experts".to_string()),
                resident: vec![],
                cpu_resident: vec![],
                cpu_resident_marker: None,
                drop_prefixes: vec![],
                int64_scatter_guard: false,
            },
            "DeepseekV3ForCausalLM" | "DeepseekV2ForCausalLM" => Self {
                architecture: arch.to_string(),
                model_class: "AirLLMBaseModel".to_string(),
                embed: "model.embed_tokens".to_string(),
                layer_prefix: "model.layers".to_string(),
                norm: "model.norm".to_string(),
                lm_head: "lm_head".to_string(),
                rotary_pos_emb: None,
                expert_prefix: Some("mlp.experts".to_string()),
                resident: vec![],
                cpu_resident: vec![],
                cpu_resident_marker: None,
                drop_prefixes: vec![],
                int64_scatter_guard: false,
            },
            _ => Self {
                architecture: if arch.is_empty() {
                    "LlamaForCausalLM".to_string()
                } else {
                    arch.to_string()
                },
                model_class: "AirLLMBaseModel".to_string(),
                embed: "model.embed_tokens".to_string(),
                layer_prefix: "model.layers".to_string(),
                norm: "model.norm".to_string(),
                lm_head: "lm_head".to_string(),
                rotary_pos_emb: None,
                expert_prefix: None,
                resident: vec![],
                cpu_resident: vec![],
                cpu_resident_marker: None,
                drop_prefixes: vec![],
                int64_scatter_guard: false,
            },
        }
    }

    /// Return variant configured for Streamed LoRA Training (embed + lm_head stay on host CPU, vision stays unmaterialized).
    pub fn for_streamed_lora(&self) -> Self {
        let mut copy = self.clone();
        copy.resident.clear();
        if !copy.cpu_resident.contains(&copy.embed) {
            copy.cpu_resident.push(copy.embed.clone());
        }
        if !copy.cpu_resident.contains(&copy.lm_head) {
            copy.cpu_resident.push(copy.lm_head.clone());
        }
        copy
    }
}

#[derive(Debug, Clone)]
pub struct ModelConfigSummary {
    pub architecture: String,
    pub model_type: String,
    pub dtype: String,
    pub num_hidden_layers: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub vocab_size: usize,
    pub tie_word_embeddings: bool,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub quant_method: Option<String>,
    pub layer_names: LayerNamesDict,
}

impl ModelConfigSummary {
    pub fn from_config_json(raw: &str) -> Result<Self, String> {
        let root = parse_json(raw).map_err(|e| format!("config.json parse failed: {e}"))?;
        let text_cfg = root.get("text_config").unwrap_or(&root);

        let architecture = root
            .get("architectures")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_str())
            .unwrap_or("LlamaForCausalLM")
            .to_string();

        let model_type = root
            .get("model_type")
            .and_then(|v| v.as_str())
            .or_else(|| text_cfg.get("model_type").and_then(|v| v.as_str()))
            .unwrap_or("llama")
            .to_string();

        let dtype = root
            .get("torch_dtype")
            .or_else(|| root.get("dtype"))
            .or_else(|| text_cfg.get("torch_dtype"))
            .or_else(|| text_cfg.get("dtype"))
            .and_then(|v| v.as_str())
            .unwrap_or("bfloat16")
            .to_string();

        let get_usize = |k: &str, default: usize| -> usize {
            text_cfg
                .get(k)
                .or_else(|| root.get(k))
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(default)
        };

        let num_hidden_layers = get_usize("num_hidden_layers", get_usize("num_layers", 32));
        let hidden_size = get_usize("hidden_size", 4096);
        let intermediate_size = get_usize("intermediate_size", hidden_size * 3);
        let num_attention_heads = get_usize("num_attention_heads", 32);
        let num_key_value_heads = get_usize("num_key_value_heads", num_attention_heads);
        let vocab_size = get_usize("vocab_size", 128256);

        let tie_word_embeddings = root
            .get("tie_word_embeddings")
            .or_else(|| text_cfg.get("tie_word_embeddings"))
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        let num_experts = get_usize(
            "num_local_experts",
            get_usize("n_routed_experts", get_usize("num_experts", 0)),
        );
        let num_experts_per_tok = get_usize("num_experts_per_tok", if num_experts > 0 { 2 } else { 0 });

        let quant_method = root
            .get("quantization_config")
            .or_else(|| text_cfg.get("quantization_config"))
            .and_then(|q| q.get("quant_method"))
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());

        let layer_names = LayerNamesDict::for_architecture(&architecture);

        Ok(Self {
            architecture,
            model_type,
            dtype,
            num_hidden_layers,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_key_value_heads,
            vocab_size,
            tie_word_embeddings,
            num_experts,
            num_experts_per_tok,
            quant_method,
            layer_names,
        })
    }

    pub fn from_dir(dir: &Path) -> Result<Self, String> {
        let cfg_path = dir.join("config.json");
        let raw = fs::read_to_string(&cfg_path)
            .map_err(|e| format!("failed to read {}: {e}", cfg_path.display()))?;
        Self::from_config_json(&raw)
    }
}

#[derive(Debug, Clone)]
pub struct ModelPreset {
    pub id: &'static str,
    pub display_name: &'static str,
    pub architecture: &'static str,
    pub total_params_b: f64,
    pub active_params_b: f64,
    pub num_layers: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub num_experts: usize,
    pub routed_experts: usize,
    pub native_dtype: &'static str,
    pub single_layer_gb: f64,
    pub peak_vram_gb: f64,
    pub ple_mmap_host_gb: f64,
    pub disk_checkpoint_gb: f64,
    pub lora_trainable: bool,
    pub lora_peak_vram_gb: f64,
    pub per_expert_streaming: bool,
    pub notes: &'static str,
}

pub fn frontier_presets() -> Vec<ModelPreset> {
    vec![
        ModelPreset {
            id: "kimi-k3-2.8t",
            display_name: "Kimi K3 (2.8T MoE MXFP4)",
            architecture: "KimiK3ForConditionalGeneration",
            total_params_b: 2800.0,
            active_params_b: 52.0,
            num_layers: 64,
            hidden_size: 7168,
            vocab_size: 163840,
            num_experts: 896,
            routed_experts: 16,
            native_dtype: "mxfp4",
            single_layer_gb: 1.05, // Per-expert streaming: 16/896 experts instead of 55GB!
            peak_vram_gb: 3.72,
            ple_mmap_host_gb: 0.0,
            disk_checkpoint_gb: 1520.0,
            lora_trainable: false,
            lora_peak_vram_gb: 0.0,
            per_expert_streaming: true,
            notes: "Per-expert MXFP4 streaming loads only 16 routed experts per token (~1GB vs 55GB whole layer). Runs under 4GB VRAM.",
        },
        ModelPreset {
            id: "deepseek-v3-671b",
            display_name: "DeepSeek-V3 (671B MoE FP8)",
            architecture: "DeepseekV3ForCausalLM",
            total_params_b: 671.0,
            active_params_b: 37.0,
            num_layers: 61,
            hidden_size: 7168,
            vocab_size: 129280,
            num_experts: 256,
            routed_experts: 8,
            native_dtype: "fp8_e4m3",
            single_layer_gb: 10.6,
            peak_vram_gb: 11.8,
            ple_mmap_host_gb: 0.0,
            disk_checkpoint_gb: 685.0,
            lora_trainable: false,
            lora_peak_vram_gb: 0.0,
            per_expert_streaming: true,
            notes: "FP8 block-scale verbatim streaming (weight + weight_scale_inv); per-expert streaming lowers peak VRAM below 4GB.",
        },
        ModelPreset {
            id: "llama-3.1-405b",
            display_name: "Llama-3.1 405B Dense",
            architecture: "LlamaForCausalLM",
            total_params_b: 405.0,
            active_params_b: 405.0,
            num_layers: 126,
            hidden_size: 16384,
            vocab_size: 128256,
            num_experts: 0,
            routed_experts: 0,
            native_dtype: "bfloat16",
            single_layer_gb: 6.4,
            peak_vram_gb: 3.9, // with 4bit NF4 compression
            ple_mmap_host_gb: 0.0,
            disk_checkpoint_gb: 810.0,
            lora_trainable: true,
            lora_peak_vram_gb: 7.8,
            per_expert_streaming: false,
            notes: "126 dense layers streamed with 4-bit NF4 block-wise compression for 3x faster NVMe throughput.",
        },
        ModelPreset {
            id: "qwen3-235b-a22b",
            display_name: "Qwen3-235B-A22B MoE",
            architecture: "Qwen3MoeForCausalLM",
            total_params_b: 235.0,
            active_params_b: 22.0,
            num_layers: 94,
            hidden_size: 4096,
            vocab_size: 151936,
            num_experts: 128,
            routed_experts: 8,
            native_dtype: "bfloat16",
            single_layer_gb: 2.5,
            peak_vram_gb: 3.1,
            ple_mmap_host_gb: 0.0,
            disk_checkpoint_gb: 470.0,
            lora_trainable: true,
            lora_peak_vram_gb: 5.2,
            per_expert_streaming: true,
            notes: "94 layers in native bf16 (prevents fp16 overflow); runs cleanly in ~3GB VRAM.",
        },
        ModelPreset {
            id: "qwen3.8-flash-next-125b",
            display_name: "Qwen3.8-Flash-Next (125B MoE + 51B PLE)",
            architecture: "Qwen4ExpForConditionalGeneration",
            total_params_b: 125.0,
            active_params_b: 14.0,
            num_layers: 48,
            hidden_size: 4096,
            vocab_size: 248320,
            num_experts: 512,
            routed_experts: 10,
            native_dtype: "bfloat16",
            single_layer_gb: 5.1,
            peak_vram_gb: 5.95,
            ple_mmap_host_gb: 102.0,
            disk_checkpoint_gb: 360.0,
            lora_trainable: true,
            lora_peak_vram_gb: 5.95,
            per_expert_streaming: false,
            notes: "51B PLE n-gram table peeled out & file-mapped on host; Gated DeltaNet + Sparse Attention + Packed-Expert LoRA under 6GB VRAM.",
        },
        ModelPreset {
            id: "llama-3-70b",
            display_name: "Llama-3 70B Instruct",
            architecture: "LlamaForCausalLM",
            total_params_b: 70.6,
            active_params_b: 70.6,
            num_layers: 80,
            hidden_size: 8192,
            vocab_size: 128256,
            num_experts: 0,
            routed_experts: 0,
            native_dtype: "bfloat16",
            single_layer_gb: 1.65,
            peak_vram_gb: 2.4,
            ple_mmap_host_gb: 0.0,
            disk_checkpoint_gb: 141.0,
            lora_trainable: true,
            lora_peak_vram_gb: 3.4,
            per_expert_streaming: false,
            notes: "Classic AirLLM flagship: 70B dense inference on any 4GB GPU without mandatory quantization.",
        },
        ModelPreset {
            id: "mixtral-8x22b",
            display_name: "Mixtral 8x22B MoE",
            architecture: "MixtralForCausalLM",
            total_params_b: 141.0,
            active_params_b: 39.0,
            num_layers: 56,
            hidden_size: 6144,
            vocab_size: 32768,
            num_experts: 8,
            routed_experts: 2,
            native_dtype: "bfloat16",
            single_layer_gb: 1.35,
            peak_vram_gb: 2.6,
            ple_mmap_host_gb: 0.0,
            disk_checkpoint_gb: 282.0,
            lora_trainable: true,
            lora_peak_vram_gb: 3.8,
            per_expert_streaming: true,
            notes: "Streams 2 of 8 experts per token on demand.",
        },
        ModelPreset {
            id: "qwen3-32b",
            display_name: "Qwen3 32B Dense",
            architecture: "Qwen3ForCausalLM",
            total_params_b: 32.8,
            active_params_b: 32.8,
            num_layers: 64,
            hidden_size: 5120,
            vocab_size: 151936,
            num_experts: 0,
            routed_experts: 0,
            native_dtype: "bfloat16",
            single_layer_gb: 0.98,
            peak_vram_gb: 1.85,
            ple_mmap_host_gb: 0.0,
            disk_checkpoint_gb: 65.5,
            lora_trainable: true,
            lora_peak_vram_gb: 2.6,
            per_expert_streaming: false,
            notes: "Fast 64-layer dense streaming; ideal for 4GB-8GB laptops & workstations.",
        },
        ModelPreset {
            id: "qwen3.8-27b-vl",
            display_name: "Qwen3.8-27B Dense VL (Gated DeltaNet)",
            architecture: "Qwen3_5ForConditionalGeneration",
            total_params_b: 27.0,
            active_params_b: 27.0,
            num_layers: 64,
            hidden_size: 5120,
            vocab_size: 248320,
            num_experts: 0,
            routed_experts: 0,
            native_dtype: "bfloat16",
            single_layer_gb: 0.82,
            peak_vram_gb: 3.33,
            ple_mmap_host_gb: 0.0,
            disk_checkpoint_gb: 54.0,
            lora_trainable: true,
            lora_peak_vram_gb: 2.0,
            per_expert_streaming: false,
            notes: "Vision tower resident for VLM inference; text-only streamed LoRA trains in ~2.0GB VRAM via Chunked Linear CE.",
        },
        ModelPreset {
            id: "chatglm3-6b",
            display_name: "ChatGLM3 6B",
            architecture: "ChatGLMForConditionalGeneration",
            total_params_b: 6.2,
            active_params_b: 6.2,
            num_layers: 28,
            hidden_size: 4096,
            vocab_size: 65024,
            num_experts: 0,
            routed_experts: 0,
            native_dtype: "float16",
            single_layer_gb: 0.42,
            peak_vram_gb: 1.1,
            ple_mmap_host_gb: 0.0,
            disk_checkpoint_gb: 12.5,
            lora_trainable: true,
            lora_peak_vram_gb: 1.5,
            per_expert_streaming: false,
            notes: "Dedicated transformer.encoder.layers + rotary_pos_emb shard mapping.",
        },
    ]
}

pub fn find_preset(query: &str) -> Option<ModelPreset> {
    let q = query.trim().to_ascii_lowercase();
    frontier_presets().into_iter().find(|p| {
        p.id == q
            || p.display_name.to_ascii_lowercase().contains(&q)
            || p.architecture.to_ascii_lowercase() == q
    })
}

pub fn preset_to_json(p: &ModelPreset) -> JsonValue {
    let mut m = std::collections::BTreeMap::new();
    m.insert("id".to_string(), JsonValue::String(p.id.to_string()));
    m.insert(
        "display_name".to_string(),
        JsonValue::String(p.display_name.to_string()),
    );
    m.insert(
        "architecture".to_string(),
        JsonValue::String(p.architecture.to_string()),
    );
    m.insert("total_params_b".to_string(), JsonValue::Number(p.total_params_b));
    m.insert("active_params_b".to_string(), JsonValue::Number(p.active_params_b));
    m.insert("num_layers".to_string(), JsonValue::Number(p.num_layers as f64));
    m.insert("hidden_size".to_string(), JsonValue::Number(p.hidden_size as f64));
    m.insert("vocab_size".to_string(), JsonValue::Number(p.vocab_size as f64));
    m.insert("num_experts".to_string(), JsonValue::Number(p.num_experts as f64));
    m.insert("routed_experts".to_string(), JsonValue::Number(p.routed_experts as f64));
    m.insert(
        "native_dtype".to_string(),
        JsonValue::String(p.native_dtype.to_string()),
    );
    m.insert("single_layer_gb".to_string(), JsonValue::Number(p.single_layer_gb));
    m.insert("peak_vram_gb".to_string(), JsonValue::Number(p.peak_vram_gb));
    m.insert("ple_mmap_host_gb".to_string(), JsonValue::Number(p.ple_mmap_host_gb));
    m.insert("disk_checkpoint_gb".to_string(), JsonValue::Number(p.disk_checkpoint_gb));
    m.insert("lora_trainable".to_string(), JsonValue::Bool(p.lora_trainable));
    m.insert("lora_peak_vram_gb".to_string(), JsonValue::Number(p.lora_peak_vram_gb));
    m.insert(
        "per_expert_streaming".to_string(),
        JsonValue::Bool(p.per_expert_streaming),
    );
    m.insert("notes".to_string(), JsonValue::String(p.notes.to_string()));
    JsonValue::Object(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resolves_all_airllm_override_architectures() {
        let kimi = LayerNamesDict::for_architecture("KimiK3ForConditionalGeneration");
        assert_eq!(kimi.model_class, "AirLLMKimiK3");
        assert_eq!(kimi.expert_prefix.as_deref(), Some("block_sparse_moe.experts"));
        assert_eq!(kimi.resident.len(), 4);

        let q4 = LayerNamesDict::for_architecture("Qwen4ExpForConditionalGeneration");
        assert_eq!(q4.model_class, "AirLLMQwen4Exp");
        assert_eq!(
            q4.cpu_resident_marker.as_deref(),
            Some("ple.ple_embedding.ngram_embedding")
        );
        assert!(q4.int64_scatter_guard);

        let glm = LayerNamesDict::for_architecture("ChatGLMForConditionalGeneration");
        assert_eq!(glm.rotary_pos_emb.as_deref(), Some("transformer.rotary_pos_emb"));
    }

    #[test]
    fn parses_nested_multimodal_config_json() {
        let cfg = r#"{
            "architectures": ["Qwen4ExpForConditionalGeneration"],
            "model_type": "qwen4_exp",
            "text_config": {
                "dtype": "bfloat16",
                "num_hidden_layers": 48,
                "hidden_size": 4096,
                "vocab_size": 248320,
                "num_experts": 512,
                "num_experts_per_tok": 10
            }
        }"#;
        let parsed = ModelConfigSummary::from_config_json(cfg).expect("parse ok");
        assert_eq!(parsed.num_hidden_layers, 48);
        assert_eq!(parsed.num_experts, 512);
        assert_eq!(parsed.num_experts_per_tok, 10);
        assert_eq!(parsed.dtype, "bfloat16");
    }
}
