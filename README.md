# PhaseVeil

**Covert-channel · audio-spectrum steganography · statistical steganalysis engine**

> **Developer:** Arush Shakya · [agiprogrammer@gmail.com](mailto:agiprogrammer@gmail.com)
>
> *Build, not Bought.*

PhaseVeil is a research-grade, fully deterministic Python engine for

1. hiding **AES-256-GCM encrypted** payloads in the **phase spectrum** of 16-bit
   PCM `.wav` files (classic *phase coding*, no LSB byte tricks),
2. transmitting hidden bits over **network inter-packet delays (IPD)** on
   loopback UDP/TCP sockets, and
3. **mathematically detecting** both covert channels with zero AI/ML - just
   Fourier analysis, distributional statistics and information-theoretic tests.

```
  $ phaseveil --version
  phaseveil 1.0.0
```

---

## Authorised-lab disclaimer (read this first)

**PhaseVeil is an educational and authorised-research tool.** It is intended for
defensive research, steganalysis methodology development, teaching, and testing
systems you own or have *explicit written permission* to test. Covert channels
are a well-known class of data-exfiltration and C2 technique; the mitigations
studied here are exactly those deployed by real defenders.

* Do **not** deploy the network channel on any network you do not own or lack
  written authorisation to test.
* The network channel is restricted by default to `127.0.0.1` loopback and uses
  only ordinary unprivileged UDP/TCP sockets: no raw packets, no `CAP_NET_ADMIN`,
  no packet crafting, no privilege escalation.
* No AI/ML models, no telemetry, no cloud APIs, no phone-home: every transform is
  pure, local, deterministic mathematics.
* The authors accept no liability for misuse.

---

## Architecture

```
                    ┌─────────────────────────────────────────────────┐
                    │                 phaseveil                       │
                    │                 (CLI entry)                     │
                    └───────┬──────────────┬──────────────┬───────────┘
                            │              │              │
        ┌───────────────────┼──────────────┼──────────────┼───────────────────┐
        │                   │              │              │                   │
        ▼                   ▼              ▼              ▼                   ▼
 ┌─────────────┐   ┌──────────────┐  ┌───────────┐  ┌────────────┐     ┌───────────┐
 │   crypto    │   │   payload    │  │audio_stego│  │  timing_   │     │ detector  │
 │ AES-256-GCM │──▶│ frame + RS   │─▶│ phase     │  │  channel   │     │ zero-ML   │
 │ zlib +      │   │ (255, 207)   │  │ coding    │  │ IPD codec  │     │ statistics│
 │ PBKDF2      │   │ CRC32        │  │           │  │ + sockets  │     │           │
 └─────────────┘   └──────────────┘  └─────┬─────┘  └─────┬──────┘     └─────┬─────┘
        │                                  │              │                  │
        │            ┌─────────────────────┘              │                  │
        │            │                                    │                  │
        │            ▼                                    ▼                  │
        │     ┌─────────────┐                     ┌───────────────┐          │
        │     │ STEGO .wav  │                     │ loopback      │          │
        │     │ 16-bit PCM  │                     │ UDP/TCP IATs  │          │
        │     └──────┬──────┘                     └───────┬───────┘          │
        │            │                                    │                  │
        │            └──────────────┬─────────────────────┘                  │
        │                           │                                        │
        │                           ▼                                        │
        │                  ┌─────────────────┐                               │
        └─────────────────▶│ phase-vector /  │◀──────────────────────────────┘
                           │ packet trace    │        (both channels feed the
                           └─────────────────┘         SAME statistical engine)
```

### Channel 1 - audio phase spectrum

```
 cover.wav ─► read 16-bit PCM ─► split into N=2048-sample segments
                                        │
              ┌─────────────────────────┼────────────────────────────┐
              │  select mid-band bins   │                            │
              │  (|X| in [0.3,1.0]x band median, dead-band 3%)       │
              │            ▼                                         │
              │   chunk of K=4 segments:                             │
              │   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐        │
              │   │ seg j0 │►│ seg j1 │►│ seg j2 │►│ seg j3 │        │
              │   │ carrier│ │ delta  │ │ delta  │ │ delta  │        │
              │   │ 8-PSK  │ │ copy   │ │ copy   │ │ copy   │        │
              │   └────────┘ └────────┘ └────────┘ └────────┘        │
              │   phases at 92 used bins → 2πv/8 (v = 3 payload bits)│
              └─────────────────────────┬────────────────────────────┘
                                        ▼
                          irfft → round to int16 → stego.wav
                                        │
                                        ▼
                    SNR 33–35 dB · PSNR 44–46 dB · ~180 B/s
```

### Channel 2 - inter-packet delays (loopback only)

```
 bytes ─► crypto blob ─► frame+RS ─► bitstream
                                     │
   preamble 1010… │ data bits        │
        │         ▼                  ▼
        │   gap = 20 ms (bit 0) or 60 ms (bit 1)
        │         │
        ▼         ▼
  ┌─────────────────────────────┐        ┌─────────────────────────────┐
  │ sender (perf_counter, spin) │  UDP   │ receiver (chronological     │
  │ pkt0 ─gap1─ pkt1 ─gap2─ pkt2│───────▶│ clock), records timestamps, │
  └─────────────────────────────┘        │127.0.0.1  decodes adaptively│
                                         └─────────────────────────────┘
```

### Detector pipeline (identical for both channels)

```
   stego.wav or trace.json
        │
        ▼
   ┌───────────────────────┐   audio ──► per-carrier grid z-scores + phase entropy
   │  feature extraction   │   timing ─► 2-means separation, IAT entropy,
   │  (deterministic)      │             lag-1 autocorrelation, KS fit
   └──────────┬────────────┘
              ▼
   ┌───────────────────────┐
   │ weighted sum [0,1]    │   fixed weights, no learning, no randomness
   └──────────┬────────────┘
              ▼
   ┌───────────────────────┐
   │ score ≥ 0.5 ?         │──▶ SUSPICIOUS (report every sub-score)
   │                       │───▶ CLEAN
   └───────────────────────┘
```

---

## Mathematical overview

### 1 · Phase coding (why no LSB)

Least-significant-bit replacement perturbs *energy* and is trivially detected by
χ² and RS-style tests. Phase coding perturbs only the **argument** of selected
frequency bins, so the power spectrum - what a spectrogram, a compressor or an
AAC encoder reacts to - is preserved bit-for-bit.

Splitting the signal into `N`-sample segments `x_j[n]` with
`X_j[k] = |X_j[k]| e^{i phi_j[k]}`:

* **carrier segment** `j0` of each chunk: the phase of selected bin `k` is
  snapped to the 8-PSK constellation

  ```
  Phi_j0[k] = 2*pi*v / M ,   v ∈ {0..M-1},   M = 8  →  bits = v in 3-bit binary
  ```

  and the conjugate bin `N-k` follows from `X[N-k] = conj(X[k])` (real input).

* **follower segments** preserve the inter-segment phase difference:

  ```
  Phi_j[k] = Phi_{j-1}[k] + (phi_j[k] - phi_{j-1}[k]),  j = j0+1 .. j0+K-1
  ⟹  Phi_j[k] - Phi_{j-1}[k] = phi_j[k] - phi_{j-1}[k]
  ```

  i.e. the *relative* phase evolution, which is what the ear and the spectrogram
  actually notice, is copied through unchanged; only the carrier's absolute
  phase is quantised.

* **Reconstruction:** `X'_j[k] = |X_j[k]| e^{i Phi_j[k]}`, then `ifft`, round to
  int16, clip.

**Energy-aware bin selection.** The error energy injected at bin `k` is
`|X_j[k]|² · 4 sin²(Δ/2) / N` (Parseval) - independent of how many bits the
constellation carries. Therefore the cheapest bins to re-phase are the *weak*
ones, and the few *strong* harmonics are the expensive ones. PhaseVeil keeps bins
whose per-file median magnitude lies in `[0.3, 1.0] × band median` and discards a
3 % dead-band around each boundary so embed and extract always agree exactly on
the bin set even though int16 quantisation perturbs magnitudes by ~1e-5. Skipping
the strong harmonics is what lifts SNR above 30 dB.

**Why M=8 and not M=2.** With M levels each modified bin carries
`log2(M)` bits for the same injected energy, so 8-PSK hides 3x the payload of
2-PSK at equal distortion. (The spec's `±pi/2` two-level scheme is the M=2
special case; `+pi/2` remains a constellation point for every even M.)

**16-bit quantisation safety.** Rounding to int16 injects white noise with
variance `1/12` LSB² per sample, i.e. `sigma = 1/sqrt(12)·sqrt(N) ≈ 13` LSB per
complex FFT bin. Phase error at bin `k` is `sigma_phi ≈ sigma/|X_k|`, and any bin
above `20·sigma` (≈260 LSB) keeps `sigma_phi < 0.05 rad`, far inside the
`±pi/M = ±0.39 rad` decision region. Bins *below* that floor (e.g. inside a
near-silent passage) are marked as Reed-Solomon **erasures**, which the FEC
repairs for free.

### 2 · Inter-packet-delay coding

Bit `0` → gap of `bit0_ms` (20 ms), bit `1` → `bit1_ms` (60 ms). The 40 ms
symbol separation versus ±8 ms worst-case OS jitter keeps the decision boundary
(40 ms) ≥ 4× further from the nominal symbols than the jitter envelope.
Decoding is *adaptive*: the threshold is scanned over candidates, chosen by
preamble match (16 alternating bits), then refined to the midpoint of the two
cluster means estimated from the preamble itself and clipped into the middle
30 % of the observed gap - so it works for any symbol pair without assuming the
listener's config matches the sender's.

### 3 · Statistical steganalysis

**Audio:** the phases of the selected bins in every *carrier* segment form a
uniform sample on `[0,2π)` for natural audio. PhaseVeil measures

* `z_max` — per carrier, the count of phases within `π/64` of a `π/4` grid point
  is Binomial(`n_bins`, `2a/grid`) under the null; standardising it gives a
  z-score whose maximum over carriers grows like `sqrt(2 ln n)` for clean audio
  (~3) but reaches ~25 for a coded carrier;
* phase-histogram entropy (secondary);
* reported descriptors: mean grid distance, `±π/2` clustering ratio.

**Timing:** 2-means separation ratio `R = (μ_hi-μ_lo)/sqrt(σ_lo²+σ_hi²)`
(unimodal traffic ≈ 1.2, a two-symbol channel ≈ 5-80), Shannon entropy of the
ms-quantised IAT histogram (tight two-symbol traffic ≈ 1 bit vs ~4 bits for
Poisson), lag-1 autocorrelation (run structure), and a Kolmogorov–Smirnov test
against the better of the MLE exponential/normal marginals. The score is a fixed
weighted sum:

```
score = 0.45·bimodal + 0.25·entropy + 0.20·lag1 + 0.10·KS   (audio: 0.7·z + 0.3·entropy)
verdict = SUSPICIOUS iff score ≥ 0.5
```

---

## Chosen parameters

| Layer    | Parameter             | Value                                            | Rationale                                                            |
|----------|-----------------------|--------------------------------------------------|----------------------------------------------------------------------|
| Audio    | FFT segment `N`       | 2048                                             | 46 ms hop, 21.5 Hz bins @44.1 kHz; `sigma/|X|` phase noise ~1e-3 rad |
| Audio    | embedding band        | 1000–6000 Hz                                     | mid-band: strong harmonics, no LF rumble / HF hiss                   |
| Audio    | constellation `M`     | 8 (3 bits/bin)                                   | 3x payload per unit injected energy vs 2-PSK                         |
| Audio    | chunk `K`             | 4 segments                                       | data carrier + 3 delta-preserving followers                          |
| Audio    | bin window            | `[0.3, 1.0] ×` band median, 3 % dead-band        | energy-aware selection, stable bin sets                              |
| Audio    | boundary ladder       | 7 multipliers, 0.2 % stability margin            | a bin sitting within 0.03 % of a selection boundary (int16 rounding scale) would otherwise shift the whole bit layout; the embedder picks a *stable* boundary and the extractor tries the same ladder                                   |
| Audio    | reliability floor     | `max(1 %, 20 × σ_q)`                             | erasures only where the phase is truly lost                          |
| FEC      | Reed–Solomon          | RS(255, 191), nsym 64                            | 32 byte errors or 64 erasures per codeword                           |
| FEC      | small-payload mode    | 1 codeword of `frame + parity`                   | ~2x shorter streams for short payloads                               |
| Crypto   | AES-256-GCM           | 200 000 PBKDF2 iterations, 16 B salt, 12 B nonce | authenticated, nonce-reuse-free                                      |
| Timing   | `bit0_ms` / `bit1_ms` | 20 / 60 ms                                       | 40 ms gap vs ±8 ms jitter = 5× margin                                |
| Timing   | preamble              | 16 alternating bits                              | sync without assuming absolute clocks                                |
| Timing   | idle timeout          | 1500 ms                                          | clean listener shutdown                                              |
| Detector | verdict threshold     | 0.50                                             | calibrated on synthetic TP/FP sweep                                  |

---

## Measured results

`phaseveil benchmark` (deterministic, synthetic 3 s multi-harmonic covers):

```
[1/3] Audio phase-coding round-trip fidelity
  plaintext      SNR 33.69 dB   PSNR 44.65 dB   embedded 143 B   capacity 552 B
  AES-256-GCM    SNR 32.58 dB   PSNR 43.53 dB   embedded 191 B   capacity 552 B
  capacity:  1 s → 172 B   2 s → 356 B   3 s → 588 B   5 s → 926 B   (44.1 kHz)
[2/3] Timing codec: 12/12 jitter trials (±3–8 ms) recover, ±8 ms worst case PASS
[3/3] Detector: audio TP 6/6, FP 0/6 · timing TP 6/6, Poisson FP 0/6, Gaussian FP 0/6
  clean audio score 0.000 vs covert 0.700 · clean Poisson 0.10 vs covert 0.55
peak RSS: 111.2 MB  (budget 150 MB)
```

Robustness sweep (`27/27` lossless round-trips over 9 cover classes x 3 seeds:
harmonic stacks, square-wave AM "buzz", near-silent ambient, inharmonic bell
partials, pure tone + noise, speech-like content with real pauses, multi-note
synthetic music, noise bursts with a quiet middle, amplitude tremolo): SNR
9.4–37.6 dB depending on how much usable mid-band energy the cover has, 0
erasure-induced failures, and every covert sample detected.

`pytest`: **136 passed** in ~95 s (includes live loopback UDP/TCP exchanges).

---

## Hardware & efficiency footprint

| Metric                                          | Value                                                               |
|-------------------------------------------------|---------------------------------------------------------------------|
| Peak RAM (full benchmark, all three subsystems) | **111.6 MB** (budget 150 MB)                                        |
| RAM, audio-only workflow (CLI `audio hide`)     | ~65 MB (`scipy.stats` loaded lazily)                                |
| CPU                                             | dual-core x86-64 Linux; single-threaded (no BLAS threads needed)    |
| Benchmark wall-clock                            | 2.3 s (3 s covers)                                                  |
| Test-suite wall-clock                           | ~70 s                                                               |
| Repository media                                | **0 bytes of binary audio** - all WAVs are synthesised in memory    |
| Dependencies                                    | `numpy`, `scipy`, `cryptography`, `reedsolo` (+ `pytest` for tests) |

Memory engineering notes:

* the phase spectrum is processed **one chunk at a time** (`K × N` complex), so
  long files never materialise a full-file FFT matrix;
* only the *embedding band* magnitudes are retained for bin selection
  (float32, `O(segments × band)`);
* `scipy.stats` (~50 MB RSS) is imported **lazily**, only when a timing trace is
  analysed.

---

## Installation

**One command, no virtualenv** (recommended - this is the verified path):

```bash
pip install --user --break-system-packages -e .
phaseveil --version
```

`--user` installs into `~/.local/lib/.../site-packages` (never touches the
system directories), `--break-system-packages` bypasses Debian/Kali's PEP 668
"externally managed environment" refusal, and `-e .` keeps the package pointing
at this source tree - so edits to `src/` take effect immediately without
reinstalling. If `~/.local/bin` is not on your `PATH`, add
`export PATH="$HOME/.local/bin:$PATH"` to your shell profile, or invoke the
entry point directly:

```bash
python3 -m phaseveil.cli --version
```

`numpy`, `scipy` and `cryptography` are usually already present on a modern
Linux image; only `reedsolo` (~30 kB, pure Python) and `pytest` (dev only) are
installed by the command above.

**If you cannot install anything at all**, the package also runs straight from
the source tree - a `conftest.py` at the repo root puts `src/` on `sys.path`:

```bash
# CLI
PYTHONPATH=src python3 -m phaseveil.cli audio hide --in cover.wav --out stego.wav --text "hi"

# test suite - no install step whatsoever
python3 -m pytest
```

**If you *do* want full dependency isolation**, `pipx` gives you the same
`phaseveil` command with its own environment (created and managed outside the
project, but never in your way):

```bash
pipx install -e .
```

Verified on this machine with both the system Python 3.14 and an Anaconda
Python 3.13 - 136 tests pass, `phaseveil benchmark` reports peak RSS 106-111 MB.

---

## CLI usage

```bash
# ---- audio phase steganography -------------------------------------------
phaseveil audio hide     --in cover.wav --out stego.wav --text "hello world"
phaseveil audio hide     --in cover.wav --out stego.wav --file secret.bin \
                         --passphrase "lab secret"
phaseveil audio extract  --in stego.wav [--passphrase "lab secret"] [--out payload.bin]

# ---- network inter-packet-delay covert channel (loopback!) ---------------
phaseveil timing send    --host 127.0.0.1 --port 9999 --text "hello" \
                         [--passphrase PASS] [--delay-ms 20,60] [--tcp]
phaseveil timing listen  --host 127.0.0.1 --port 9999 [--passphrase PASS] \
                         [--out payload.bin] [--save-trace trace.json] [--tcp]

# ---- deterministic statistical steganalysis ------------------------------
phaseveil analyze audio  --in sample.wav [--baseline clean_cover.wav]
phaseveil analyze timing --trace trace.json

# ---- benchmark (capacity, SNR, jitter, detector accuracy, peak RAM) ------
phaseveil benchmark [--json bench.json]
```

Example session:

```bash
$ phaseveil audio hide --in cover.wav --out stego.wav \
      --text "PhaseVeil demo: hidden in the phase spectrum." --passphrase "lab-demo-key"
phaseveil audio hide  value
--------------------  ----------------------
cover                 cover.wav
stego                 stego.wav
payload bytes         45
mode                  AES-256-GCM + zlib
FEC                   RS(255,207)
embedded bytes        142
capacity bytes        552
capacity used         25.7%
used bins             92
chunks                5/16
SNR                   33.80 dB
PSNR                  44.76 dB
erasures              0 bits (0.0%)

$ phaseveil audio extract --in stego.wav --passphrase "lab-demo-key"
PhaseVeil demo: hidden in the phase spectrum.
(payload 99 B, flags 0x03, 1 RS codewords)
```

---

## Project layout

```
src/phaseveil/
  config.py          every parameter of the scheme (single source of truth)
  crypto.py          AES-256-GCM + zlib + PBKDF2 blob sealing
  payload.py         [MAGIC|VER_FLAGS|LEN|PAYLOAD|CRC32] + RS(255,207) FEC
  audio_stego.py     M-PSK phase coding with delta preservation + metrics
  timing_channel.py  pure IPD codec + live 127.0.0.1 UDP/TCP I/O + traces
  detector.py        statistical steganalysis (phase z-tests, IAT tests)
  benchmark.py       synthetic capacity/SNR/jitter/detector benchmark
  cli.py             argparse CLI
tests/               133 tests, synthetic fixtures only, no binary media
```

---

## Limitations (honest)

* The audio detector is *targeted*: it knows the candidate scheme's geometry
  (segment length, band, chunk structure). Untargeted "any-steganography"
  detection is a harder problem and out of scope.
* A cover whose mid-band phases are *already* quantised onto the same grid -
  e.g. a synthetic signal whose harmonics all start at the same phase, which
  literally *is* phase coding - is a degenerate case that no deterministic
  phase statistic can separate (see the asserting test in
  `tests/test_detector.py::test_degenerate_in_phase_harmonic_cover_is_flagged`).
  Use `--baseline` for differential analysis on unusual covers.
* SNR tracks how much usable mid-band energy the cover has: a spectrally rich
  cover reaches 30+ dB while a nearly silent one can fall below 20 dB.  The
  capacity/bits-per-selection adapt automatically, and the erasure + FEC layer
  guarantees lossless round-trips in every tested case.
* The timing channel assumes a stable clock between sender and receiver (always
  true on loopback; cross-machine links would need periodic re-synchronisation).
* Payload size is bounded by the RS(255, ·) block structure (≈191 KB per
  codeword chain); huge payloads simply need proportionally longer covers.

## License

MIT — see [LICENSE](LICENSE). Educational/research use only; see the disclaimer above.
