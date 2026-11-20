//! Hardened Localhost HTTP/REST, OpenAI-Compatible & MCP Gateway Server.
//! Pure Safe Rust (`std::net::TcpListener`), 127.0.0.1 default binding,
//! SEC-04 10 MiB body cap + token-bucket rate limiter, SEC-05 security headers,
//! and embedded offline AAA Precision Cockpit (`assets/cockpit.html`).

use crate::VERSION;
use crate::bench::run_full_benchmark;
use crate::catalog::{find_preset, frontier_presets, preset_to_json};
use crate::dream_rsi::{compile_ide_prompt, dream_evolution_proposals, execute_2click_proposal};
use crate::hal::{CompressionMode, HardwareProfile, build_streaming_plan};
use crate::json::{JsonValue, parse_json};
use crate::lora::estimate_streamed_lora_budget;
use crate::mcp::handle_mcp_request;
use crate::security::{MAX_HTTP_BODY_BYTES, TokenBucketLimiter, security_headers};
use std::collections::BTreeMap;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Duration;

const COCKPIT_HTML: &str = include_str!("../assets/cockpit.html");

pub fn run_http_server(bind_addr: &str, project_dir: PathBuf) -> Result<(), String> {
    let listener = TcpListener::bind(bind_addr)
        .map_err(|e| format!("failed to bind HTTP server on {bind_addr}: {e}"))?;
    let limiter = Arc::new(Mutex::new(TokenBucketLimiter::new(120.0, 40.0)));
    eprintln!(
        "[NALA-airllm-rs v{VERSION}] Cockpit & API listening on http://{bind_addr}/ (100% Pure Safe Rust)"
    );

    for stream in listener.incoming() {
        let Ok(mut sock) = stream else {
            continue;
        };
        let peer_ip = sock
            .peer_addr()
            .map(|a| a.ip().to_string())
            .unwrap_or_else(|_| "127.0.0.1".to_string());
        let allowed = limiter
            .lock()
            .map(|mut l| l.allow(&peer_ip))
            .unwrap_or(true);
        if !allowed {
            let _ = write_response(
                &mut sock,
                429,
                "Too Many Requests",
                "application/json; charset=utf-8",
                r#"{"error":"SEC-04 rate limit exceeded"}"#.as_bytes(),
            );
            continue;
        }
        let _ = sock.set_read_timeout(Some(Duration::from_secs(10)));
        let _ = sock.set_write_timeout(Some(Duration::from_secs(10)));
        handle_connection(&mut sock, &project_dir);
    }
    Ok(())
}

fn handle_connection(sock: &mut TcpStream, project_dir: &PathBuf) {
    let mut header_buf = Vec::with_capacity(4096);
    let mut tmp = [0u8; 2048];
    let header_end;
    loop {
        let Ok(n) = sock.read(&mut tmp) else {
            return;
        };
        if n == 0 {
            return;
        }
        header_buf.extend_from_slice(&tmp[..n]);
        if let Some(pos) = find_subslice(&header_buf, b"\r\n\r\n") {
            header_end = pos + 4;
            break;
        }
        if header_buf.len() > 64 * 1024 {
            let _ = write_response(sock, 431, "Header Too Large", "text/plain", b"Header too large");
            return;
        }
    }

    let header_text = String::from_utf8_lossy(&header_buf[..header_end]);
    let mut lines = header_text.lines();
    let req_line = lines.next().unwrap_or("");
    let mut parts = req_line.split_whitespace();
    let method = parts.next().unwrap_or("GET");
    let raw_path = parts.next().unwrap_or("/");
    let path = raw_path.split('?').next().unwrap_or("/");

    let mut content_length: usize = 0;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            if k.trim().eq_ignore_ascii_case("content-length") {
                content_length = v.trim().parse::<usize>().unwrap_or(0);
            }
        }
    }

    // SEC-04: Strict 10 MiB payload cap
    if content_length > MAX_HTTP_BODY_BYTES {
        let _ = write_response(
            sock,
            413,
            "Payload Too Large",
            "application/json; charset=utf-8",
            r#"{"error":"SEC-04 request body exceeds 10 MiB limit"}"#.as_bytes(),
        );
        return;
    }

    let mut body = header_buf[header_end..].to_vec();
    while body.len() < content_length {
        let Ok(n) = sock.read(&mut tmp) else {
            break;
        };
        if n == 0 {
            break;
        }
        body.extend_from_slice(&tmp[..n]);
    }
    body.truncate(content_length);
    let body_str = String::from_utf8_lossy(&body);

    match (method, path) {
        ("GET", "/") | ("GET", "/cockpit") => {
            let _ = write_response(
                sock,
                200,
                "OK",
                "text/html; charset=utf-8",
                COCKPIT_HTML.as_bytes(),
            );
        }
        ("GET", "/health") => {
            let hw = HardwareProfile::detect();
            let mut m = BTreeMap::new();
            m.insert("status".to_string(), JsonValue::String("ok".to_string()));
            m.insert("engine".to_string(), JsonValue::String("nala-airllm-rs".to_string()));
            m.insert("version".to_string(), JsonValue::String(VERSION.to_string()));
            m.insert("zero_python".to_string(), JsonValue::Bool(true));
            m.insert("safe_rust".to_string(), JsonValue::Bool(true));
            m.insert("hardware".to_string(), hw.to_json());
            let json = JsonValue::Object(m).to_json_string();
            let _ = write_response(sock, 200, "OK", "application/json; charset=utf-8", json.as_bytes());
        }
        ("GET", "/api/presets") => {
            let arr = JsonValue::Array(frontier_presets().iter().map(preset_to_json).collect());
            let json = arr.to_json_string();
            let _ = write_response(sock, 200, "OK", "application/json; charset=utf-8", json.as_bytes());
        }
        ("GET", "/v1/models") => {
            let data: Vec<JsonValue> = frontier_presets()
                .iter()
                .map(|p| {
                    let mut m = BTreeMap::new();
                    m.insert("id".to_string(), JsonValue::String(p.id.to_string()));
                    m.insert("object".to_string(), JsonValue::String("model".to_string()));
                    m.insert("owned_by".to_string(), JsonValue::String("nala-airllm-rs".to_string()));
                    JsonValue::Object(m)
                })
                .collect();
            let mut root = BTreeMap::new();
            root.insert("object".to_string(), JsonValue::String("list".to_string()));
            root.insert("data".to_string(), JsonValue::Array(data));
            let json = JsonValue::Object(root).to_json_string();
            let _ = write_response(sock, 200, "OK", "application/json; charset=utf-8", json.as_bytes());
        }
        ("POST", "/api/plan") => {
            let parsed = parse_json(&body_str).unwrap_or(JsonValue::Object(BTreeMap::new()));
            let model = parsed
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("kimi-k3-2.8t");
            let comp_s = parsed
                .get("compression")
                .and_then(|v| v.as_str())
                .unwrap_or("4bit");
            let comp = CompressionMode::parse(comp_s).unwrap_or(CompressionMode::Nf4);
            let prefetch = parsed
                .get("prefetching")
                .and_then(|v| v.as_bool())
                .unwrap_or(true);
            let del = parsed
                .get("delete_original")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            let hw = if let Some(vram) = parsed.get("vram_gb").and_then(|v| v.as_f64()) {
                HardwareProfile::simulated(vram, 64.0, "cuda")
            } else {
                HardwareProfile::detect()
            };
            let plan = build_streaming_plan(model, comp, prefetch, del, &hw).to_json();
            let json = plan.to_json_string();
            let _ = write_response(sock, 200, "OK", "application/json; charset=utf-8", json.as_bytes());
        }
        ("POST", "/api/lora/plan") => {
            let parsed = parse_json(&body_str).unwrap_or(JsonValue::Object(BTreeMap::new()));
            let model = parsed
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("qwen3.8-flash-next-125b");
            let p = find_preset(model).unwrap_or_else(|| frontier_presets()[4].clone());
            let rank = parsed.get("rank").and_then(|v| v.as_u64()).unwrap_or(16) as usize;
            let seq = parsed.get("seq_len").and_then(|v| v.as_u64()).unwrap_or(512) as usize;
            let chunk = parsed
                .get("ce_chunk_size")
                .and_then(|v| v.as_u64())
                .unwrap_or(4096) as usize;
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
            )
            .to_json();
            let json = est.to_json_string();
            let _ = write_response(sock, 200, "OK", "application/json; charset=utf-8", json.as_bytes());
        }
        ("GET", "/api/bench") | ("POST", "/api/bench") => {
            let json = run_full_benchmark().to_json_string();
            let _ = write_response(sock, 200, "OK", "application/json; charset=utf-8", json.as_bytes());
        }
        ("GET", "/api/dream/proposals") => {
            let arr = JsonValue::Array(
                dream_evolution_proposals()
                    .iter()
                    .map(|p| p.to_json())
                    .collect(),
            );
            let json = arr.to_json_string();
            let _ = write_response(sock, 200, "OK", "application/json; charset=utf-8", json.as_bytes());
        }
        ("POST", "/api/dream/apply") => {
            let parsed = parse_json(&body_str).unwrap_or(JsonValue::Object(BTreeMap::new()));
            let pid = parsed
                .get("proposal_id")
                .and_then(|v| v.as_str())
                .unwrap_or("");
            let confirm = parsed
                .get("confirm_step2")
                .and_then(|v| v.as_bool())
                .unwrap_or(false);
            match execute_2click_proposal(project_dir, pid, confirm) {
                Ok(val) => {
                    let json = val.to_json_string();
                    let _ = write_response(sock, 200, "OK", "application/json; charset=utf-8", json.as_bytes());
                }
                Err(e) => {
                    let mut m = BTreeMap::new();
                    m.insert("error".to_string(), JsonValue::String(e));
                    let json = JsonValue::Object(m).to_json_string();
                    let _ = write_response(sock, 400, "Bad Request", "application/json; charset=utf-8", json.as_bytes());
                }
            }
        }
        ("POST", "/api/dream/compile-prompt") => {
            let parsed = parse_json(&body_str).unwrap_or(JsonValue::Object(BTreeMap::new()));
            let ide = parsed
                .get("ide")
                .and_then(|v| v.as_str())
                .unwrap_or("Antigravity");
            let pid = parsed
                .get("proposal_id")
                .and_then(|v| v.as_str())
                .unwrap_or("AIRLLM-RSI-01-DIRECT-NVME-PREFETCH-RING");
            match compile_ide_prompt(ide, pid) {
                Ok(prompt) => {
                    let mut m = BTreeMap::new();
                    m.insert("ide".to_string(), JsonValue::String(ide.to_string()));
                    m.insert("proposal_id".to_string(), JsonValue::String(pid.to_string()));
                    m.insert("prompt".to_string(), JsonValue::String(prompt));
                    let json = JsonValue::Object(m).to_json_string();
                    let _ = write_response(sock, 200, "OK", "application/json; charset=utf-8", json.as_bytes());
                }
                Err(e) => {
                    let mut m = BTreeMap::new();
                    m.insert("error".to_string(), JsonValue::String(e));
                    let json = JsonValue::Object(m).to_json_string();
                    let _ = write_response(sock, 400, "Bad Request", "application/json; charset=utf-8", json.as_bytes());
                }
            }
        }
        ("POST", "/v1/chat/completions") => {
            let parsed = parse_json(&body_str).unwrap_or(JsonValue::Object(BTreeMap::new()));
            let model = parsed
                .get("model")
                .and_then(|v| v.as_str())
                .unwrap_or("kimi-k3-2.8t");
            let hw = HardwareProfile::detect();
            let plan = build_streaming_plan(model, CompressionMode::Nf4, true, false, &hw);
            let reply = format!(
                "[NALA-airllm-rs v{VERSION}] Ready for layer-wise streaming of {} (Peak VRAM: {:.2} GB, Single-Layer I/O: {:.2} GB, Bottleneck: {}). Point `nala-airllm-rs infer --dir <splitted_model>` at a local sharded checkpoint to stream tokens.",
                plan.display_name, plan.peak_vram_gb, plan.single_layer_disk_gb, plan.roofline_bottleneck
            );
            let mut msg = BTreeMap::new();
            msg.insert("role".to_string(), JsonValue::String("assistant".to_string()));
            msg.insert("content".to_string(), JsonValue::String(reply));
            let mut choice = BTreeMap::new();
            choice.insert("index".to_string(), JsonValue::Number(0.0));
            choice.insert("message".to_string(), JsonValue::Object(msg));
            choice.insert("finish_reason".to_string(), JsonValue::String("stop".to_string()));
            let mut root = BTreeMap::new();
            root.insert("id".to_string(), JsonValue::String("chatcmpl-nala-airllm-rs".to_string()));
            root.insert("object".to_string(), JsonValue::String("chat.completion".to_string()));
            root.insert("model".to_string(), JsonValue::String(model.to_string()));
            root.insert("choices".to_string(), JsonValue::Array(vec![JsonValue::Object(choice)]));
            let json = JsonValue::Object(root).to_json_string();
            let _ = write_response(sock, 200, "OK", "application/json; charset=utf-8", json.as_bytes());
        }
        ("POST", "/mcp") => {
            let resp = handle_mcp_request(&body_str, VERSION);
            let _ = write_response(sock, 200, "OK", "application/json; charset=utf-8", resp.as_bytes());
        }
        _ => {
            let _ = write_response(
                sock,
                404,
                "Not Found",
                "application/json; charset=utf-8",
                r#"{"error":"route not found"}"#.as_bytes(),
            );
        }
    }
}

fn write_response(
    sock: &mut TcpStream,
    code: u16,
    reason: &str,
    content_type: &str,
    body: &[u8],
) -> std::io::Result<()> {
    let sec = security_headers(VERSION);
    let head = format!(
        "HTTP/1.1 {code} {reason}\r\n\
         Content-Type: {content_type}\r\n\
         Content-Length: {}\r\n\
         Connection: close\r\n\
         {sec}\r\n",
        body.len()
    );
    sock.write_all(head.as_bytes())?;
    sock.write_all(body)?;
    sock.flush()
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}
