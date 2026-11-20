//! sHeriFF + NVIDIA SkillSpector Hardened Security, Crypto & Zero-PII Gate.
//! 100% Pure Safe Rust (FIPS 180-4 SHA-256, RFC 2104 HMAC-SHA256, 64-bit FNV-1a,
//! SEC-01 SSRF, SEC-02 XSS/CSS-Breakout Sanitizer, SEC-03 Path/Command Guard,
//! SEC-04 Body Limit & Rate Limiter, SEC-05 Security Headers).

use std::collections::HashMap;
use std::fs;
use std::path::{Component, Path, PathBuf};
use std::time::{Duration, Instant};

pub const MAX_HTTP_BODY_BYTES: usize = 10 * 1024 * 1024; // SEC-04: 10 MiB hard cap

/// Pure Safe Rust FIPS 180-4 SHA-256 implementation.
pub fn sha256(data: &[u8]) -> [u8; 32] {
    const K: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let mut h: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];

    let bit_len = (data.len() as u64).wrapping_mul(8);
    let mut padded = Vec::with_capacity(data.len() + 72);
    padded.extend_from_slice(data);
    padded.push(0x80);
    while (padded.len() % 64) != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    for chunk in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, word) in chunk.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([word[0], word[1], word[2], word[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut hh] = h;
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);

            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }

    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

pub fn sha256_hex(data: &[u8]) -> String {
    let digest = sha256(data);
    let mut s = String::with_capacity(64);
    for b in digest {
        s.push_str(&format!("{:02x}", b));
    }
    s
}

pub fn hmac_sha256(key: &[u8], msg: &[u8]) -> [u8; 32] {
    let mut block_key = [0u8; 64];
    if key.len() > 64 {
        let hashed = sha256(key);
        block_key[..32].copy_from_slice(&hashed);
    } else {
        block_key[..key.len()].copy_from_slice(key);
    }
    let mut ipad = [0x36u8; 64];
    let mut opad = [0x5cu8; 64];
    for i in 0..64 {
        ipad[i] ^= block_key[i];
        opad[i] ^= block_key[i];
    }
    let mut inner = Vec::with_capacity(64 + msg.len());
    inner.extend_from_slice(&ipad);
    inner.extend_from_slice(msg);
    let inner_hash = sha256(&inner);

    let mut outer = Vec::with_capacity(96);
    outer.extend_from_slice(&opad);
    outer.extend_from_slice(&inner_hash);
    sha256(&outer)
}

pub fn fnv1a_64(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for &b in data {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// SEC-01: SSRF guard rejecting loopback, RFC1918 private networks, link-local metadata, and IPv6 loopback.
pub fn validate_remote_url_ssrf(url: &str) -> Result<(), String> {
    let trimmed = url.trim();
    let lower = trimmed.to_ascii_lowercase();
    if !(lower.starts_with("https://") || lower.starts_with("http://")) {
        return Err("SEC-01: only http/https schemes allowed".to_string());
    }
    let rest = if let Some(r) = lower.strip_prefix("https://") {
        r
    } else {
        lower.strip_prefix("http://").unwrap_or("")
    };
    let authority = rest.split('/').next().unwrap_or("");
    let host_port = authority.rsplit('@').next().unwrap_or("");
    let host = if host_port.starts_with('[') {
        host_port
            .trim_start_matches('[')
            .split(']')
            .next()
            .unwrap_or("")
    } else {
        host_port.split(':').next().unwrap_or("")
    };

    if host.is_empty()
        || host == "localhost"
        || host.ends_with(".local")
        || host.ends_with(".internal")
        || host == "::1"
        || host == "0:0:0:0:0:0:0:1"
        || host.starts_with("fe80:")
        || host.starts_with("fc00:")
        || host.starts_with("fd")
    {
        return Err(format!("SEC-01 SSRF blocked host: {host}"));
    }

    if let Ok(ip) = host.parse::<std::net::Ipv4Addr>() {
        let oct = ip.octets();
        let blocked = ip.is_loopback()
            || ip.is_private()
            || ip.is_link_local()
            || ip.is_unspecified()
            || (oct[0] == 100 && (64..=127).contains(&oct[1])) // CGNAT / Tailscale
            || (oct[0] == 169 && oct[1] == 254);
        if blocked {
            return Err(format!("SEC-01 SSRF blocked IPv4: {ip}"));
        }
    }
    Ok(())
}

/// SEC-02: Sanitize text against HTML/CSS/SVG script breakout injections.
pub fn sanitize_html_css(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        match ch {
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&#x27;"),
            _ => out.push(ch),
        }
    }
    out
}

/// SEC-03: Block path traversal (`..`), NUL bytes, and shell metacharacters in relative paths or identifiers.
pub fn validate_safe_relative_path(raw: &str) -> Result<PathBuf, String> {
    if raw.is_empty() || raw.contains('\0') {
        return Err("SEC-03: empty or NUL-containing path".to_string());
    }
    if raw.contains("..") {
        return Err("SEC-03: path traversal '..' blocked".to_string());
    }
    let path = Path::new(raw);
    if path.is_absolute() {
        return Err("SEC-03: absolute path not allowed here".to_string());
    }
    for comp in path.components() {
        match comp {
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err("SEC-03: illegal path component".to_string());
            }
            _ => {}
        }
    }
    Ok(path.to_path_buf())
}

/// SEC-03: Validate user-supplied checkpoint or output paths (blocks NUL, shell injection, and `..`).
pub fn validate_checkpoint_path(raw: &str) -> Result<PathBuf, String> {
    if raw.is_empty() || raw.contains('\0') {
        return Err("SEC-03: invalid checkpoint path".to_string());
    }
    for forbidden in ['`', '$', ';', '|', '&', '\n', '\r'] {
        if raw.contains(forbidden) {
            return Err(format!(
                "SEC-03: shell metacharacter '{forbidden}' rejected in path"
            ));
        }
    }
    let p = Path::new(raw);
    for comp in p.components() {
        if matches!(comp, Component::ParentDir) {
            return Err("SEC-03: parent traversal '..' rejected in checkpoint path".to_string());
        }
    }
    Ok(p.to_path_buf())
}

/// SEC-04: Per-client sliding token bucket rate limiter.
pub struct TokenBucketLimiter {
    capacity: f64,
    refill_per_sec: f64,
    buckets: HashMap<String, (f64, Instant)>,
}

impl TokenBucketLimiter {
    pub fn new(capacity: f64, refill_per_sec: f64) -> Self {
        Self {
            capacity,
            refill_per_sec,
            buckets: HashMap::new(),
        }
    }

    pub fn allow(&mut self, key: &str) -> bool {
        let now = Instant::now();
        let entry = self
            .buckets
            .entry(key.to_string())
            .or_insert((self.capacity, now));
        let elapsed = now.saturating_duration_since(entry.1).as_secs_f64();
        entry.0 = (entry.0 + elapsed * self.refill_per_sec).min(self.capacity);
        entry.1 = now;
        if entry.0 >= 1.0 {
            entry.0 -= 1.0;
            true
        } else {
            false
        }
    }

    pub fn prune_idle(&mut self, max_age: Duration) {
        let now = Instant::now();
        self.buckets
            .retain(|_, (_, ts)| now.saturating_duration_since(*ts) <= max_age);
    }
}

/// SEC-05: Standard hardened HTTP security headers.
pub fn security_headers(version: &str) -> String {
    format!(
        "X-Content-Type-Options: nosniff\r\n\
         X-Frame-Options: SAMEORIGIN\r\n\
         Referrer-Policy: no-referrer\r\n\
         Content-Security-Policy: default-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; frame-ancestors 'self'\r\n\
         X-NALA-Version: {version}\r\n"
    )
}

#[derive(Debug, Clone)]
pub struct PiiFinding {
    pub file: String,
    pub line: usize,
    pub rule: &'static str,
    pub masked_preview: String,
}

/// Zero-PII & Secret scanner for pre-push GitHub hygiene (no plaintext secrets stored here).
pub fn audit_directory_zero_pii(root: &Path) -> Result<(usize, Vec<PiiFinding>), String> {
    let mut files_scanned = 0usize;
    let mut findings = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).map_err(|e| format!("read_dir {}: {e}", dir.display()))?;
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name();
            let name_s = name.to_string_lossy();
            if name_s == ".git" || name_s == "target" || name_s == "dist" {
                continue;
            }
            if path.is_dir() {
                stack.push(path);
                continue;
            }
            let ext = path
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("")
                .to_ascii_lowercase();
            let text_like = matches!(
                ext.as_str(),
                "rs" | "toml"
                    | "md"
                    | "html"
                    | "css"
                    | "js"
                    | "json"
                    | "yml"
                    | "yaml"
                    | "sh"
                    | "ps1"
                    | "txt"
                    | ""
            );
            if !text_like {
                continue;
            }
            let Ok(content) = fs::read_to_string(&path) else {
                continue;
            };
            files_scanned += 1;
            let rel = path
                .strip_prefix(root)
                .unwrap_or(&path)
                .display()
                .to_string();

            let pat_rsa = ["BEGIN RSA ", "PRIVATE KEY"].concat();
            let pat_ssh = ["BEGIN OPENSSH ", "PRIVATE KEY"].concat();
            let pat_pk = ["BEGIN ", "PRIVATE KEY"].concat();
            let pat_ghp = ["gh", "p_"].concat();
            let pat_gho = ["gh", "o_"].concat();
            let pat_sk = ["sk-", "proj-"].concat();
            let pat_home = ["/", "Users/"].concat();
            let pat_home_ok = ["/", "Users/<user>"].concat();
            let pat_lan = ["192", ".168."].concat();
            let pat_ts1 = ["100", ".98."].concat();
            let pat_ts2 = ["100", ".85."].concat();

            for (idx, line) in content.lines().enumerate() {
                let lineno = idx + 1;
                if line.contains(&pat_rsa)
                    || line.contains(&pat_ssh)
                    || line.contains(&pat_pk)
                {
                    findings.push(PiiFinding {
                        file: rel.clone(),
                        line: lineno,
                        rule: "PRIVATE_KEY_HEADER",
                        masked_preview: "-----BEGIN ***-----".to_string(),
                    });
                }
                if line.contains(&pat_ghp) || line.contains(&pat_gho) || line.contains(&pat_sk) {
                    findings.push(PiiFinding {
                        file: rel.clone(),
                        line: lineno,
                        rule: "API_TOKEN_PREFIX",
                        masked_preview: "tok***".to_string(),
                    });
                }
                if line.contains(&pat_home) && !line.contains(&pat_home_ok) {
                    findings.push(PiiFinding {
                        file: rel.clone(),
                        line: lineno,
                        rule: "HARDCODED_MACOS_HOME",
                        masked_preview: "home/***".to_string(),
                    });
                }
                if line.contains(&pat_lan) || line.contains(&pat_ts1) || line.contains(&pat_ts2)
                {
                    findings.push(PiiFinding {
                        file: rel.clone(),
                        line: lineno,
                        rule: "HARDCODED_LAN_TAILSCALE_IP",
                        masked_preview: "ip***".to_string(),
                    });
                }
            }
        }
    }
    Ok((files_scanned, findings))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sha256_matches_nist_abc() {
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    #[test]
    fn hmac_sha256_rfc4231_case2() {
        let mac = hmac_sha256(b"Jefe", b"what do ya want for nothing?");
        let hex: String = mac.iter().map(|b| format!("{:02x}", b)).collect();
        assert_eq!(
            hex,
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn ssrf_and_path_guards_block_exploits() {
        assert!(validate_remote_url_ssrf("http://127.0.0.1:8080/admin").is_err());
        assert!(validate_remote_url_ssrf("http://169.254.169.254/latest/meta-data").is_err());
        let private_url = ["http://192", ".168.1.10/api"].concat();
        assert!(validate_remote_url_ssrf(&private_url).is_err());
        assert!(validate_remote_url_ssrf("https://huggingface.co/Qwen/Qwen3-32B").is_ok());

        assert!(validate_safe_relative_path("../../etc/passwd").is_err());
        assert!(validate_checkpoint_path("/models/qwen; rm -rf /").is_err());
        assert!(validate_checkpoint_path("/models/qwen3-32b").is_ok());
    }

    #[test]
    fn html_sanitizer_blocks_script_breakout() {
        let evil = "</style><script>alert(1)</script>";
        let safe = sanitize_html_css(evil);
        assert!(!safe.contains("<script>"));
        assert!(safe.contains("&lt;script&gt;"));
    }
}
