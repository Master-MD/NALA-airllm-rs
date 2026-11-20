# Security Policy (`sHeriFF` Hardening Standard)

## Supported Versions

| Version | Status |
| :--- | :--- |
| `4.0.x-nala.6.6.6` | Actively supported & security-audited |

## Built-In Security Guarantees
- **Memory Safety:** `#![forbid(unsafe_code)]` enforced at crate root (`src/lib.rs` & `src/main.rs`).
- **Zero Supply-Chain Surface:** `0` external crate dependencies (`std` only), Zero-Python, Zero-Node.js.
- **5-Layer `sHeriFF` Defense (`src/security.rs`):**
  - `SEC-01`: SSRF guard blocking loopback, RFC1918 private networks, CGNAT, and cloud metadata IPs.
  - `SEC-02`: HTML/CSS/SVG script-breakout sanitizer.
  - `SEC-03`: Path-traversal (`..`), NUL-byte, and shell metacharacter validation on all checkpoint/shard paths.
  - `SEC-04`: Strict `10 MiB` HTTP request body limit (`413 Payload Too Large`) + per-client token-bucket rate limiter (`429 Too Many Requests`).
  - `SEC-05`: Hardened HTTP headers (`X-Content-Type-Options: nosniff`, `X-Frame-Options: SAMEORIGIN`, `Referrer-Policy: no-referrer`, strict `CSP`, `X-NALA-Version`) and `127.0.0.1` loopback default binding.
- **Zero-PII Gate:** Every commit & release passes `nala-airllm-rs audit-pii .` and cryptographic `SHA256SUMS.txt` verification.

## Reporting a Vulnerability
Please report security issues privately via GitHub Security Advisories or open a non-exploitable tracking issue in `Master-MD/NALA-airllm-rs`. Never include secrets, private keys, or personal data in public reports.
