# Security Policy

PhaseVeil is an **authorized-research** tool for studying covert channels,
steganography, and steganalysis. It is designed to be safe to *inspect*: it
performs no network access other than the loopback sockets it explicitly opens
for the timing channel, collects no telemetry, and phones home to nothing.

## Supported versions

Only the latest released line receives security fixes.

| Version | Supported |
| ------- | --------- |
| 1.0.x   | ✅ Yes    |
| < 1.0   | ❌ No     |

## What counts as a security issue

In scope:

- **Cryptographic failures** — a wrong passphrase accepted, a tampered
  ciphertext authenticated, AES-GCM tag verification bypassed, key derivation
  weakened (for example by reducing PBKDF2 iterations), a reuse of salt/nonce
  that breaks the AEAD guarantees.
- **Silent corruption** — Reed-Solomon/FEC paths that return *wrong* payload
  bytes without raising, CRC32 checks that can be bypassed, or extraction that
  succeeds with corrupted data.
- **Memory/DoS robustness** — unbounded memory growth or crashes on crafted
  `.wav` input or adversarial packet traces (the project budget is 150 MB peak
  RSS for legitimate inputs).
- **Determinism breaks** — a code path whose output depends on unseeded
  randomness, wall-clock time, or filesystem ordering where the documented
  contract promises deterministic output.
- **Network scope violations** — any code path that opens a non-loopback socket,
  uses raw sockets, requires elevated privileges, or contacts a remote host
  other than the user-specified `--host`.
- **Command/code execution** — injection via CLI arguments, file paths, or
  payload bytes.

Out of scope (these are the tool's documented purpose, not defects):

- "PhaseVeil can hide data" / "PhaseVeil can transmit covertly" — that is the
  stated research purpose, released under an MIT license with an authorized-use
  disclaimer.
- **Detectability by a strong adversary.** No steganographic scheme defeats an
  adversary who knows the exact parameters and holds the cover; PhaseVeil's
  detectors demonstrate exactly this. See the README "Limitations" section.
- Documented design limits: the timing channel assumes a stable clock between
  endpoints (always true on loopback), degenerate in-phase harmonic covers,
  and payload capacity bounded by the RS block structure.
- Missing hardening features that are not part of the documented contract.
- Vulnerabilities in upstream dependencies (`numpy`, `scipy`, `cryptography`,
  `reedsolo`) — please report those to their maintainers.

## Reporting a vulnerability

**Preferred:** use GitHub *private* vulnerability reporting —
<https://github.com/agiprogrammerch/PhaseVeil/security/advisories/new>.
Reports are visible only to the maintainers and can be fixed in a private fork
before disclosure.

**Alternative:** email **agiprogrammer@gmail.com** with the subject line
`[PhaseVeil security] <short summary>`. Please include:

1. affected component and version,
2. a minimal, deterministic reproduction (inputs + commands),
3. the impact you believe it has,
4. any suggested fix.

Please **do not** open a public issue for a suspected vulnerability.

## Response targets

| Stage | Target |
| ----- | ------ |
| Acknowledgement | 72 hours |
| Triage / severity decision | 7 days |
| Fix or mitigation for confirmed issues | 30 days, coordinated with you |

We follow **coordinated disclosure**: once a fix is available we will credit you
(if you wish) and publish the advisory, and we ask that you allow a 90-day
window from your report.

## Safe harbor

Good-faith security research conducted under this policy is authorized. We will
not pursue or support legal action against researchers who: investigate only
this project on systems they own or are authorized to test, avoid privacy
violations and destruction of data, and give us a reasonable window to fix what
they find.

## Reminder

Use PhaseVeil only on networks and systems you own or have written permission
to test. The network channel defaults to `127.0.0.1` loopback with unprivileged
UDP/TCP sockets precisely so that it *cannot* touch a network it should not.