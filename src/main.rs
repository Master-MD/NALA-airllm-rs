#![forbid(unsafe_code)]

//! NALA-airllm-rs CLI Entrypoint.
//! 100% Pure Safe Rust (Edition 2024, 0 External Crates, Zero-Python, Zero-Telemetry).

use nala_airllm_rs::VERSION;
use nala_airllm_rs::bench::run_full_benchmark;
use nala_airllm_rs::catalog::{LayerNamesDict, ModelConfigSummary, find_preset, frontier_presets, preset_to_json};
use nala_airllm_rs::dream_rsi::{
    compile_ide_prompt, create_ncsp_snapshot, dream_evolution_proposals, execute_2click_proposal,
};
use nala_airllm_rs::hal::{CompressionMode, HardwareProfile, build_streaming_plan};
use nala_airllm_rs::json::JsonValue;
use nala_airllm_rs::lora::{estimate_streamed_lora_budget, load_sft_records};
use nala_airllm_rs::mcp::run_mcp_stdio;
use nala_airllm_rs::safetensors::{
    gather_ngram_rows, read_safetensors_header, split_safetensors_checkpoint,
};
use nala_airllm_rs::security::{audit_directory_zero_pii, validate_checkpoint_path};
use nala_airllm_rs::server::run_http_server;
use nala_airllm_rs::streamer::run_streamed_forward;
use std::collections::BTreeMap;
use std::env;
use std::path::{Path, PathBuf};
use std::process;

fn main() {
    if let Err(err) = run() {
        eprintln!("[nala-airllm-rs] ERROR: {err}");
        process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() || matches!(args[0].as_str(), "-h" | "--help" | "help") {
        print_help();
        return Ok(());
    }

    match args[0].as_str() {
        "version" | "--version" | "-V" => {
            println!("nala-airllm-rs v{VERSION} (Rust Edition 2024, #![forbid(unsafe_code)], 0 Crates, Zero-Python)");
            Ok(())
        }
        "presets" => {
            let arr = JsonValue::Array(frontier_presets().iter().map(preset_to_json).collect());
            println!("{}", arr.to_json_pretty());
            Ok(())
        }
        "detect" => {
            let target = args
                .get(1)
                .ok_or_else(|| "usage: nala-airllm-rs detect <model_dir_or_architecture>".to_string())?;
            let p = Path::new(target);
            if p.is_dir() && p.join("config.json").is_file() {
                let safe = validate_checkpoint_path(target)?;
                let cfg = ModelConfigSummary::from_dir(&safe)?;
                let mut m = BTreeMap::new();
                m.insert("architecture".to_string(), JsonValue::String(cfg.architecture));
                m.insert("model_type".to_string(), JsonValue::String(cfg.model_type));
                m.insert("model_class".to_string(), JsonValue::String(cfg.layer_names.model_class));
                m.insert("dtype".to_string(), JsonValue::String(cfg.dtype));
                m.insert("num_hidden_layers".to_string(), JsonValue::Number(cfg.num_hidden_layers as f64));
                m.insert("hidden_size".to_string(), JsonValue::Number(cfg.hidden_size as f64));
                m.insert("vocab_size".to_string(), JsonValue::Number(cfg.vocab_size as f64));
                m.insert("tie_word_embeddings".to_string(), JsonValue::Bool(cfg.tie_word_embeddings));
                m.insert("num_experts".to_string(), JsonValue::Number(cfg.num_experts as f64));
                m.insert("embed".to_string(), JsonValue::String(cfg.layer_names.embed));
                m.insert("layer_prefix".to_string(), JsonValue::String(cfg.layer_names.layer_prefix));
                m.insert("norm".to_string(), JsonValue::String(cfg.layer_names.norm));
                m.insert("lm_head".to_string(), JsonValue::String(cfg.layer_names.lm_head));
                println!("{}", JsonValue::Object(m).to_json_pretty());
            } else {
                let ldict = LayerNamesDict::for_architecture(target);
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
                m.insert("int64_scatter_guard".to_string(), JsonValue::Bool(ldict.int64_scatter_guard));
                println!("{}", JsonValue::Object(m).to_json_pretty());
            }
            Ok(())
        }
        "plan" => {
            let model = flag_value(&args, "--model").unwrap_or_else(|| "kimi-k3-2.8t".to_string());
            let comp_s = flag_value(&args, "--compression").unwrap_or_else(|| "4bit".to_string());
            let comp = CompressionMode::parse(&comp_s)?;
            let prefetch = !has_flag(&args, "--no-prefetch");
            let delete_orig = has_flag(&args, "--delete-original");
            let hw = if let Some(v) = flag_value(&args, "--simulate-vram-gb") {
                let vram: f64 = v.parse().map_err(|_| "invalid --simulate-vram-gb".to_string())?;
                let ram: f64 = flag_value(&args, "--simulate-ram-gb")
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(64.0);
                let backend = flag_value(&args, "--backend").unwrap_or_else(|| "cuda".to_string());
                HardwareProfile::simulated(vram, ram, &backend)
            } else {
                HardwareProfile::detect()
            };
            let plan = build_streaming_plan(&model, comp, prefetch, delete_orig, &hw);
            println!("{}", plan.to_json().to_json_pretty());
            Ok(())
        }
        "split" => {
            let dir = flag_value(&args, "--dir")
                .ok_or_else(|| "usage: nala-airllm-rs split --dir <checkpoint_dir> [--out <dir>] [--arch <Arch>] [--compression none|8bit|4bit] [--delete-original]".to_string())?;
            let safe_dir = validate_checkpoint_path(&dir)?;
            let out_dir = match flag_value(&args, "--out") {
                Some(o) => Some(validate_checkpoint_path(&o)?),
                None => None,
            };
            let arch = flag_value(&args, "--arch").unwrap_or_else(|| {
                ModelConfigSummary::from_dir(&safe_dir)
                    .map(|c| c.architecture)
                    .unwrap_or_else(|_| "LlamaForCausalLM".to_string())
            });
            let comp = CompressionMode::parse(
                &flag_value(&args, "--compression").unwrap_or_else(|| "none".to_string()),
            )?;
            let delete_orig = has_flag(&args, "--delete-original");
            let ldict = LayerNamesDict::for_architecture(&arch);
            let rep = split_safetensors_checkpoint(
                &safe_dir,
                out_dir.as_deref(),
                &ldict,
                comp,
                delete_orig,
            )?;
            let mut m = BTreeMap::new();
            m.insert("architecture".to_string(), JsonValue::String(arch));
            m.insert("compression".to_string(), JsonValue::String(comp.as_str().to_string()));
            m.insert("output_dir".to_string(), JsonValue::String(rep.output_dir.display().to_string()));
            m.insert("total_layers_written".to_string(), JsonValue::Number(rep.total_layers_written as f64));
            m.insert("passthrough_hardlinked".to_string(), JsonValue::Number(rep.passthrough_hardlinked as f64));
            m.insert("ngram_tables_persisted".to_string(), JsonValue::Number(rep.ngram_tables_persisted as f64));
            m.insert("total_bytes_written".to_string(), JsonValue::Number(rep.total_bytes_written as f64));
            println!("{}", JsonValue::Object(m).to_json_pretty());
            Ok(())
        }
        "inspect" => {
            let path = args
                .get(1)
                .ok_or_else(|| "usage: nala-airllm-rs inspect <file.safetensors>".to_string())?;
            let safe = validate_checkpoint_path(path)?;
            let hdr = read_safetensors_header(&safe)?;
            let mut m = BTreeMap::new();
            m.insert("file".to_string(), JsonValue::String(safe.display().to_string()));
            m.insert("header_len".to_string(), JsonValue::Number(hdr.header_len as f64));
            m.insert("tensor_count".to_string(), JsonValue::Number(hdr.tensors.len() as f64));
            let payload_bytes: u64 = hdr.tensors.values().map(|t| t.byte_len()).sum();
            m.insert("payload_bytes".to_string(), JsonValue::Number(payload_bytes as f64));
            let sample: Vec<JsonValue> = hdr
                .tensors
                .values()
                .take(20)
                .map(|t| {
                    let mut tm = BTreeMap::new();
                    tm.insert("name".to_string(), JsonValue::String(t.name.clone()));
                    tm.insert("dtype".to_string(), JsonValue::String(t.dtype.clone()));
                    tm.insert("bytes".to_string(), JsonValue::Number(t.byte_len() as f64));
                    JsonValue::Object(tm)
                })
                .collect();
            m.insert("sample_tensors".to_string(), JsonValue::Array(sample));
            println!("{}", JsonValue::Object(m).to_json_pretty());
            Ok(())
        }
        "ngram-gather" => {
            let dir = flag_value(&args, "--dir").ok_or_else(|| "missing --dir".to_string())?;
            let layer = flag_value(&args, "--layer").ok_or_else(|| "missing --layer".to_string())?;
            let rows_s = flag_value(&args, "--rows").unwrap_or_else(|| "0".to_string());
            let indices: Vec<usize> = rows_s
                .split(',')
                .filter_map(|s| s.trim().parse::<usize>().ok())
                .collect();
            let safe = validate_checkpoint_path(&dir)?;
            let (vals, cols) = gather_ngram_rows(&safe, &layer, &indices)?;
            println!(
                "{{\"rows_gathered\":{},\"cols\":{},\"first_val\":{}}}",
                indices.len(),
                cols,
                vals.first().copied().unwrap_or(0.0)
            );
            Ok(())
        }
        "infer" => {
            let dir = flag_value(&args, "--dir")
                .ok_or_else(|| "usage: nala-airllm-rs infer --dir <splitted_model_dir> --layers N --hidden H --vocab V [--tokens 1,2,3] [--arch LlamaForCausalLM]".to_string())?;
            let safe = validate_checkpoint_path(&dir)?;
            let arch = flag_value(&args, "--arch").unwrap_or_else(|| "LlamaForCausalLM".to_string());
            let layers: usize = flag_value(&args, "--layers")
                .and_then(|v| v.parse().ok())
                .unwrap_or(2);
            let hidden: usize = flag_value(&args, "--hidden")
                .and_then(|v| v.parse().ok())
                .unwrap_or(64);
            let vocab: usize = flag_value(&args, "--vocab")
                .and_then(|v| v.parse().ok())
                .unwrap_or(128);
            let tokens: Vec<usize> = flag_value(&args, "--tokens")
                .unwrap_or_else(|| "1,2,3".to_string())
                .split(',')
                .filter_map(|s| s.trim().parse().ok())
                .collect();
            let prefetch = !has_flag(&args, "--no-prefetch");
            let ldict = LayerNamesDict::for_architecture(&arch);
            let res = run_streamed_forward(&safe, &ldict, &tokens, layers, hidden, vocab, prefetch)?;
            let mut m = BTreeMap::new();
            m.insert("next_token_id".to_string(), JsonValue::Number(res.next_token_id as f64));
            m.insert("profiler".to_string(), res.profiler.to_json());
            println!("{}", JsonValue::Object(m).to_json_pretty());
            Ok(())
        }
        "lora-plan" => {
            let model = flag_value(&args, "--model")
                .unwrap_or_else(|| "qwen3.8-flash-next-125b".to_string());
            let p = find_preset(&model).unwrap_or_else(|| frontier_presets()[4].clone());
            let rank: usize = flag_value(&args, "--rank")
                .and_then(|v| v.parse().ok())
                .unwrap_or(16);
            let seq: usize = flag_value(&args, "--seq-len")
                .and_then(|v| v.parse().ok())
                .unwrap_or(512);
            let chunk: usize = flag_value(&args, "--ce-chunk")
                .and_then(|v| v.parse().ok())
                .unwrap_or(4096);
            let flash = p.architecture == "Qwen4ExpForConditionalGeneration";
            let est = estimate_streamed_lora_budget(
                p.id,
                p.num_layers,
                p.hidden_size,
                p.single_layer_gb,
                rank,
                seq,
                chunk,
                flash,
            );
            println!("{}", est.to_json().to_json_pretty());
            Ok(())
        }
        "lora-data" => {
            let path = args
                .get(1)
                .ok_or_else(|| "usage: nala-airllm-rs lora-data <dataset.jsonl|json|txt>".to_string())?;
            let safe = validate_checkpoint_path(path)?;
            let recs = load_sft_records(&safe)?;
            println!(
                "{{\"records\":{},\"masked_records\":{}}}",
                recs.len(),
                recs.iter().filter(|r| r.mask_prefix.is_some()).count()
            );
            Ok(())
        }
        "bench" => {
            println!("{}", run_full_benchmark().to_json_pretty());
            Ok(())
        }
        "doctor" => {
            let hw = HardwareProfile::detect();
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let (scanned, findings) = audit_directory_zero_pii(&cwd)?;
            let mut m = BTreeMap::new();
            m.insert("version".to_string(), JsonValue::String(VERSION.to_string()));
            m.insert("safe_rust".to_string(), JsonValue::Bool(true));
            m.insert("zero_python".to_string(), JsonValue::Bool(true));
            m.insert("hardware".to_string(), hw.to_json());
            m.insert("zero_pii_files_scanned".to_string(), JsonValue::Number(scanned as f64));
            m.insert("zero_pii_findings".to_string(), JsonValue::Number(findings.len() as f64));
            println!("{}", JsonValue::Object(m).to_json_pretty());
            Ok(())
        }
        "audit-pii" => {
            let dir = args
                .get(1)
                .map(PathBuf::from)
                .unwrap_or_else(|| env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
            let (scanned, findings) = audit_directory_zero_pii(&dir)?;
            println!("sHeriFF Zero-PII Audit: {scanned} files scanned, {} findings.", findings.len());
            for f in &findings {
                println!("  FAIL [{}] {}:{} -> {}", f.rule, f.file, f.line, f.masked_preview);
            }
            if !findings.is_empty() {
                return Err(format!("Zero-PII gate failed with {} finding(s)", findings.len()));
            }
            Ok(())
        }
        "dream" => {
            if let Some(ide) = flag_value(&args, "--compile-prompt") {
                let pid = flag_value(&args, "--proposal")
                    .unwrap_or_else(|| "AIRLLM-RSI-01-DIRECT-NVME-PREFETCH-RING".to_string());
                println!("{}", compile_ide_prompt(&ide, &pid)?);
                return Ok(());
            }
            if let Some(pid) = flag_value(&args, "--apply") {
                let confirm = has_flag(&args, "--confirm-step2");
                let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                let res = execute_2click_proposal(&cwd, &pid, confirm)?;
                println!("{}", res.to_json_pretty());
                return Ok(());
            }
            let arr = JsonValue::Array(
                dream_evolution_proposals()
                    .iter()
                    .map(|p| p.to_json())
                    .collect(),
            );
            println!("{}", arr.to_json_pretty());
            Ok(())
        }
        "snapshot" => {
            let tag = args.get(1).map(|s| s.as_str()).unwrap_or("manual");
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            let snap = create_ncsp_snapshot(&cwd, tag)?;
            println!("NCSP Snapshot created: {}", snap.display());
            println!("Rollback script      : {}", snap.join("rollback.sh").display());
            Ok(())
        }
        "mcp" => run_mcp_stdio(VERSION),
        "serve" => {
            let addr = flag_value(&args, "--bind").unwrap_or_else(|| "127.0.0.1:18794".to_string());
            let cwd = env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
            run_http_server(&addr, cwd)
        }
        other => Err(format!(
            "unknown subcommand '{other}'. Run `nala-airllm-rs --help` for available commands."
        )),
    }
}

fn flag_value(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == flag {
            return args.get(i + 1).cloned();
        }
        if let Some(rest) = args[i].strip_prefix(&format!("{flag}=")) {
            return Some(rest.to_string());
        }
    }
    None
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn print_help() {
    println!(
        "NALA-airllm-rs v{VERSION} — 100% Pure Safe Rust Layer-Wise 4GB GPU LLM Engine
Zero-Python | Zero-Node.js | 0 External Crates | #![forbid(unsafe_code)] | Cross-OS

USAGE:
  nala-airllm-rs <COMMAND> [OPTIONS]

COMMANDS:
  plan         Compute 4GB GPU layer-streaming Roofline plan (--model, --compression, --simulate-vram-gb)
  detect       Detect Hugging Face config.json or architecture LayerNamesDict overrides
  presets      List all 10 built-in Frontier Model Presets (Kimi K3 2.8T .. ChatGLM3 6B)
  split        Split Safetensors checkpoint into per-layer shards (+ hardlink passthrough, PLE mmap, NF4/INT8)
  inspect      Zero-payload header inspection of a .safetensors shard
  ngram-gather Direct file-seek gather from a persisted PLE .mmap table
  infer        Run double-buffered layer-wise & per-expert streamed forward pass
  lora-plan    Estimate Streamed LoRA + Chunked Linear Cross-Entropy VRAM budget
  lora-data    Validate and count SFT training records (.jsonl / .json / .txt)
  bench        Run live Pure-Rust hardware micro-benchmarks (NF4, INT8, MXFP4 GEMV, Chunked CE)
  doctor       Hardware HAL, memory, backend, and Zero-PII self-check
  audit-pii    Run sHeriFF Zero-PII & secret scanner on repository
  dream        List Dream/Pearls/REA/RSI proposals, 2-Click apply, or compile Multi-IDE prompts
  snapshot     Create atomic NCSP v2.5 SHA-256 snapshot + rollback.sh
  serve        Start hardened 127.0.0.1 HTTP/OpenAI/MCP Server + AAA Cockpit UI (default :18794)
  mcp          Start native JSON-RPC 2.0 MCP Server over stdio (12 tools)
  version      Show SSOT version"
    );
}
