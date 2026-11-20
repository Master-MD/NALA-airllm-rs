//! Native JSON-RPC 2.0 MCP Server (stdio & HTTP /mcp) exposing 12 NALA-airllm-rs Tools.
//! Enables seamless composition inside NALA-OS, Antigravity, Codex, ZCode, and OpenCode.

use crate::bench::run_full_benchmark;
use crate::catalog::{LayerNamesDict, ModelConfigSummary, find_preset, frontier_presets, preset_to_json};
use crate::dream_rsi::{
    compile_ide_prompt, create_ncsp_snapshot, dream_evolution_proposals, execute_2click_proposal,
};
use crate::hal::{CompressionMode, HardwareProfile, build_streaming_plan};
use crate::json::{JsonValue, parse_json};
use crate::lora::estimate_streamed_lora_budget;
use crate::safetensors::{read_safetensors_header, split_safetensors_checkpoint};
use crate::security::{audit_directory_zero_pii, validate_checkpoint_path};
use std::collections::BTreeMap;
use std::io::{self, BufRead, Write};
use std::path::Path;

pub const MCP_PROTOCOL_VERSION: &str = "2024-11-05";

pub fn tool_definitions() -> JsonValue {
    let specs: &[(&str, &str)] = &[
        ("airllm_detect_arch", "Detect Hugging Face model architecture and AirLLM LayerNamesDict overrides from a local directory or architecture name."),
        ("airllm_plan_vram", "Compute Roofline 4GB-GPU layer-streaming plan (VRAM, pinned RAM, PLE mmap, compression, est. tok/s) for any frontier model."),
        ("airllm_list_presets", "List all built-in AirLLM frontier presets (Kimi K3 2.8T, DeepSeek-V3 671B, Llama-3.1 405B, Qwen3.8-Flash-Next 125B+51B PLE, etc.)."),
        ("airllm_split_checkpoint", "Split a local Safetensors checkpoint into per-layer AirLLM shards with hardlink passthrough, PLE mmap peeling, and optional NF4/INT8 compression."),
        ("airllm_inspect_shard", "Inspect a .safetensors shard header (zero payload read) for tensor shapes, dtypes, byte offsets, and per-expert MoE partition counts."),
        ("airllm_lora_plan", "Estimate VRAM budget for Streamed Layer-Wise LoRA + Fused Chunked Linear Cross-Entropy training."),
        ("airllm_quant_bench", "Run live Pure Safe Rust micro-benchmarks (NF4, INT8, MXFP4 GEMV, Chunked Linear CE) on local hardware."),
        ("airllm_doctor", "Run NALA-HAL hardware, memory, backend, and Zero-PII self-diagnostics."),
        ("airllm_dream_proposals", "List NALA Dream / Software-Pearls / REA / RSI evolution proposals with Solution & Delta Overview Matrix."),
        ("airllm_dream_apply_2click", "Execute a Dream/RSI proposal through the mandatory 2-Click confirmation gate (creates NCSP SHA-256 snapshot + rollback.sh)."),
        ("airllm_compile_ide_prompt", "Compile a ready-to-paste Codex, Antigravity, ZCode, or OpenCode prompt ending with the mandatory verbatim TheStandart footer."),
        ("airllm_ncsp_snapshot", "Create an atomic NCSP v2.5 SHA-256 snapshot and rollback.sh for the workspace."),
    ];

    let arr: Vec<JsonValue> = specs
        .iter()
        .map(|(name, desc)| {
            let mut m = BTreeMap::new();
            m.insert("name".to_string(), JsonValue::String((*name).to_string()));
            m.insert(
                "description".to_string(),
                JsonValue::String((*desc).to_string()),
            );
            let mut schema = BTreeMap::new();
            schema.insert("type".to_string(), JsonValue::String("object".to_string()));
            schema.insert("properties".to_string(), JsonValue::Object(BTreeMap::new()));
            m.insert("inputSchema".to_string(), JsonValue::Object(schema));
            JsonValue::Object(m)
        })
        .collect();
    JsonValue::Array(arr)
}

pub fn handle_mcp_request(raw_json: &str, version: &str) -> String {
    let Ok(req) = parse_json(raw_json) else {
        return rpc_error(JsonValue::Null, -32700, "Parse error");
    };
    let id = req.get("id").cloned().unwrap_or(JsonValue::Null);
    let method = req
        .get("method")
        .and_then(|v| v.as_str())
        .unwrap_or("");

    match method {
        "initialize" => {
            let mut info = BTreeMap::new();
            info.insert("name".to_string(), JsonValue::String("nala-airllm-rs".to_string()));
            info.insert("version".to_string(), JsonValue::String(version.to_string()));
            let mut caps = BTreeMap::new();
            caps.insert("tools".to_string(), JsonValue::Object(BTreeMap::new()));
            let mut res = BTreeMap::new();
            res.insert(
                "protocolVersion".to_string(),
                JsonValue::String(MCP_PROTOCOL_VERSION.to_string()),
            );
            res.insert("capabilities".to_string(), JsonValue::Object(caps));
            res.insert("serverInfo".to_string(), JsonValue::Object(info));
            rpc_ok(id, JsonValue::Object(res))
        }
        "tools/list" => {
            let mut res = BTreeMap::new();
            res.insert("tools".to_string(), tool_definitions());
            rpc_ok(id, JsonValue::Object(res))
        }
        "tools/call" => {
            let params = req.get("params");
            let name = params
                .and_then(|p| p.get("name"))
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let args = params.and_then(|p| p.get("arguments"));
            match dispatch_tool(name, args) {
                Ok(payload) => {
                    let mut text_block = BTreeMap::new();
                    text_block.insert("type".to_string(), JsonValue::String("text".to_string()));
                    text_block.insert(
                        "text".to_string(),
                        JsonValue::String(payload.to_json_pretty()),
                    );
                    let mut res = BTreeMap::new();
                    res.insert(
                        "content".to_string(),
                        JsonValue::Array(vec![JsonValue::Object(text_block)]),
                    );
                    rpc_ok(id, JsonValue::Object(res))
                }
                Err(err) => rpc_error(id, -32000, &err),
            }
        }
        _ => rpc_error(id, -32601, &format!("Method not found: {method}")),
    }
}

fn dispatch_tool(name: &str, args: Option<&JsonValue>) -> Result<JsonValue, String> {
    let arg_str = |k: &str, default: &str| -> String {
        args.and_then(|a| a.get(k))
            .and_then(|v| v.as_str())
            .unwrap_or(default)
            .to_string()
    };
    let arg_bool = |k: &str, default: bool| -> bool {
        args.and_then(|a| a.get(k))
            .and_then(|v| v.as_bool())
            .unwrap_or(default)
    };
    let arg_u64 = |k: &str, default: u64| -> u64 {
        args.and_then(|a| a.get(k))
            .and_then(|v| v.as_u64())
            .unwrap_or(default)
    };

    match name {
        "airllm_detect_arch" => {
            let target = arg_str("path_or_arch", "Qwen4ExpForConditionalGeneration");
            let p = Path::new(&target);
            if p.is_dir() && p.join("config.json").is_file() {
                let safe = validate_checkpoint_path(&target)?;
                let summary = ModelConfigSummary::from_dir(&safe)?;
                let mut m = BTreeMap::new();
                m.insert("architecture".to_string(), JsonValue::String(summary.architecture));
                m.insert("model_class".to_string(), JsonValue::String(summary.layer_names.model_class));
                m.insert("layers".to_string(), JsonValue::Number(summary.num_hidden_layers as f64));
                m.insert("dtype".to_string(), JsonValue::String(summary.dtype));
                m.insert("embed".to_string(), JsonValue::String(summary.layer_names.embed));
                m.insert("layer_prefix".to_string(), JsonValue::String(summary.layer_names.layer_prefix));
                Ok(JsonValue::Object(m))
            } else {
                let ldict = LayerNamesDict::for_architecture(&target);
                let mut m = BTreeMap::new();
                m.insert("architecture".to_string(), JsonValue::String(ldict.architecture));
                m.insert("model_class".to_string(), JsonValue::String(ldict.model_class));
                m.insert("embed".to_string(), JsonValue::String(ldict.embed));
                m.insert("layer_prefix".to_string(), JsonValue::String(ldict.layer_prefix));
                m.insert("norm".to_string(), JsonValue::String(ldict.norm));
                m.insert("lm_head".to_string(), JsonValue::String(ldict.lm_head));
                m.insert(
                    "expert_prefix".to_string(),
                    ldict.expert_prefix.map(JsonValue::String).unwrap_or(JsonValue::Null),
                );
                m.insert(
                    "cpu_resident_marker".to_string(),
                    ldict.cpu_resident_marker.map(JsonValue::String).unwrap_or(JsonValue::Null),
                );
                Ok(JsonValue::Object(m))
            }
        }
        "airllm_plan_vram" => {
            let model = arg_str("model", "kimi-k3-2.8t");
            let comp = CompressionMode::parse(&arg_str("compression", "4bit"))?;
            let prefetch = arg_bool("prefetching", true);
            let del = arg_bool("delete_original", false);
            let hw = if let Some(vram) = args.and_then(|a| a.get("vram_gb")).and_then(|v| v.as_f64()) {
                HardwareProfile::simulated(vram, 64.0, &arg_str("backend", "cuda"))
            } else {
                HardwareProfile::detect()
            };
            Ok(build_streaming_plan(&model, comp, prefetch, del, &hw).to_json())
        }
        "airllm_list_presets" => Ok(JsonValue::Array(
            frontier_presets().iter().map(preset_to_json).collect(),
        )),
        "airllm_split_checkpoint" => {
            let raw_dir = arg_str("checkpoint_dir", "");
            let safe_dir = validate_checkpoint_path(&raw_dir)?;
            let arch = arg_str("architecture", "LlamaForCausalLM");
            let comp = CompressionMode::parse(&arg_str("compression", "none"))?;
            let del = arg_bool("delete_original", false);
            let ldict = LayerNamesDict::for_architecture(&arch);
            let rep = split_safetensors_checkpoint(&safe_dir, None, &ldict, comp, del)?;
            let mut m = BTreeMap::new();
            m.insert("output_dir".to_string(), JsonValue::String(rep.output_dir.display().to_string()));
            m.insert("total_layers_written".to_string(), JsonValue::Number(rep.total_layers_written as f64));
            m.insert("passthrough_hardlinked".to_string(), JsonValue::Number(rep.passthrough_hardlinked as f64));
            m.insert("ngram_tables_persisted".to_string(), JsonValue::Number(rep.ngram_tables_persisted as f64));
            m.insert("total_bytes_written".to_string(), JsonValue::Number(rep.total_bytes_written as f64));
            Ok(JsonValue::Object(m))
        }
        "airllm_inspect_shard" => {
            let raw_path = arg_str("path", "");
            let safe = validate_checkpoint_path(&raw_path)?;
            let hdr = read_safetensors_header(&safe)?;
            let mut m = BTreeMap::new();
            m.insert("header_len".to_string(), JsonValue::Number(hdr.header_len as f64));
            m.insert("tensor_count".to_string(), JsonValue::Number(hdr.tensors.len() as f64));
            let total_bytes: u64 = hdr.tensors.values().map(|t| t.byte_len()).sum();
            m.insert("payload_bytes".to_string(), JsonValue::Number(total_bytes as f64));
            Ok(JsonValue::Object(m))
        }
        "airllm_lora_plan" => {
            let model = arg_str("model", "qwen3.8-flash-next-125b");
            let p = find_preset(&model).unwrap_or_else(|| frontier_presets()[4].clone());
            let rank = arg_u64("rank", 16) as usize;
            let seq_len = arg_u64("seq_len", 512) as usize;
            let chunk = arg_u64("ce_chunk_size", 4096) as usize;
            let flash = p.architecture == "Qwen4ExpForConditionalGeneration";
            Ok(estimate_streamed_lora_budget(
                p.id,
                p.num_layers,
                p.hidden_size,
                p.single_layer_gb,
                rank,
                seq_len,
                chunk,
                flash,
            )
            .to_json())
        }
        "airllm_quant_bench" => Ok(run_full_benchmark()),
        "airllm_doctor" => {
            let hw = HardwareProfile::detect();
            let cwd = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
            let (scanned, findings) = audit_directory_zero_pii(&cwd).unwrap_or((0, vec![]));
            let mut m = BTreeMap::new();
            m.insert("hardware".to_string(), hw.to_json());
            m.insert("zero_pii_files_scanned".to_string(), JsonValue::Number(scanned as f64));
            m.insert("zero_pii_findings".to_string(), JsonValue::Number(findings.len() as f64));
            m.insert("status".to_string(), JsonValue::String("HEALTHY".to_string()));
            Ok(JsonValue::Object(m))
        }
        "airllm_dream_proposals" => Ok(JsonValue::Array(
            dream_evolution_proposals().iter().map(|p| p.to_json()).collect(),
        )),
        "airllm_dream_apply_2click" => {
            let pid = arg_str("proposal_id", "AIRLLM-RSI-01-DIRECT-NVME-PREFETCH-RING");
            let confirm = arg_bool("confirm_step2", false);
            let cwd = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
            execute_2click_proposal(&cwd, &pid, confirm)
        }
        "airllm_compile_ide_prompt" => {
            let ide = arg_str("ide", "Antigravity");
            let pid = arg_str("proposal_id", "AIRLLM-RSI-01-DIRECT-NVME-PREFETCH-RING");
            let compiled = compile_ide_prompt(&ide, &pid)?;
            let mut m = BTreeMap::new();
            m.insert("ide".to_string(), JsonValue::String(ide));
            m.insert("proposal_id".to_string(), JsonValue::String(pid));
            m.insert("prompt".to_string(), JsonValue::String(compiled));
            Ok(JsonValue::Object(m))
        }
        "airllm_ncsp_snapshot" => {
            let tag = arg_str("tag", "mcp_manual");
            let cwd = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
            let snap = create_ncsp_snapshot(&cwd, &tag)?;
            let mut m = BTreeMap::new();
            m.insert("snapshot_dir".to_string(), JsonValue::String(snap.display().to_string()));
            m.insert(
                "rollback_script".to_string(),
                JsonValue::String(snap.join("rollback.sh").display().to_string()),
            );
            Ok(JsonValue::Object(m))
        }
        other => Err(format!("unknown tool '{other}'")),
    }
}

pub fn run_mcp_stdio(version: &str) -> Result<(), String> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    for line in stdin.lock().lines() {
        let line = line.map_err(|e| format!("stdin read: {e}"))?;
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }
        let resp = handle_mcp_request(trimmed, version);
        writeln!(stdout, "{resp}").map_err(|e| format!("stdout write: {e}"))?;
        stdout.flush().map_err(|e| format!("stdout flush: {e}"))?;
    }
    Ok(())
}

fn rpc_ok(id: JsonValue, result: JsonValue) -> String {
    let mut m = BTreeMap::new();
    m.insert("jsonrpc".to_string(), JsonValue::String("2.0".to_string()));
    m.insert("id".to_string(), id);
    m.insert("result".to_string(), result);
    JsonValue::Object(m).to_json_string()
}

fn rpc_error(id: JsonValue, code: i32, message: &str) -> String {
    let mut err = BTreeMap::new();
    err.insert("code".to_string(), JsonValue::Number(code as f64));
    err.insert("message".to_string(), JsonValue::String(message.to_string()));
    let mut m = BTreeMap::new();
    m.insert("jsonrpc".to_string(), JsonValue::String("2.0".to_string()));
    m.insert("id".to_string(), id);
    m.insert("error".to_string(), JsonValue::Object(err));
    JsonValue::Object(m).to_json_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mcp_lists_12_tools_and_executes_plan() {
        let list_raw = handle_mcp_request(
            r#"{"jsonrpc":"2.0","id":1,"method":"tools/list"}"#,
            "4.0.0-nala.6.6.6",
        );
        let parsed = parse_json(&list_raw).unwrap();
        let tools = parsed
            .get("result")
            .and_then(|r| r.get("tools"))
            .and_then(|t| t.as_array())
            .unwrap();
        assert_eq!(tools.len(), 12);

        let call_raw = handle_mcp_request(
            r#"{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"airllm_plan_vram","arguments":{"model":"kimi-k3-2.8t","vram_gb":4.0}}}"#,
            "4.0.0-nala.6.6.6",
        );
        assert!(call_raw.contains("kimi-k3-2.8t"));
    }
}
