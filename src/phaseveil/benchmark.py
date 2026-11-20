"""Synthetic benchmark: capacity, SNR, jitter recovery, detector accuracy, RAM.

Every sample is generated **deterministically** from fixed seeds, so two runs of
``phaseveil benchmark`` report identical numbers (only wall-clock timings vary).
No binary fixtures are written to disk - all WAVs are synthesised in memory.
"""

from __future__ import annotations

import math
import resource
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import PhaseVeilConfig
from . import audio_stego, timing_channel
from .crypto import make_blob, parse_blob
from .detector import (
    VERDICT_CLEAN,
    VERDICT_SUSPICIOUS,
    _audio_feature_scores,
    analyze_timing_trace,
    phase_statistics,
)


def _audio_anomaly(samples: np.ndarray, rate: int, cfg: PhaseVeilConfig) -> float:
    """Anomaly score for an in-memory audio buffer (no temp files)."""
    metrics = phase_statistics(samples, rate, cfg.audio)
    score, _subs = _audio_feature_scores(metrics, cfg.detector, None)
    return score

# ---------------------------------------------------------------------------
# Synthetic signal generators
# ---------------------------------------------------------------------------


def synth_cover(
    seconds: float = 3.0,
    sample_rate: int = 44100,
    *,
    seed: int = 1234,
    fundamental: float = 220.0,
    harmonics: int = 24,
    noise_sigma: float = 120.0,
    quiet_patch: bool = True,
) -> Tuple[int, np.ndarray]:
    """Multi-harmonic 16-bit PCM cover (deterministic).

    Contains tapering harmonics, a faint noise floor and (optionally) a
    near-silent passage so the benchmark exercises erasure handling too.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    x = np.zeros(n, dtype=np.float64)
    for h in range(1, harmonics + 1):
        amp = 9000.0 / (h**1.1) * (0.6 + 0.4 * np.sin(2 * math.pi * 0.7 * h * t))
        if h % 5 == 0:
            amp *= 0.15
        x += amp * np.sin(2 * math.pi * fundamental * h * t + 0.3 * h)
    x += rng.normal(0.0, noise_sigma, size=n)
    if quiet_patch:
        a, b = int(1.0 * sample_rate), int(1.2 * sample_rate)
        x[a:b] *= 0.05
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak * 0.75 * 32767.0
    return sample_rate, np.clip(np.rint(x), -32768.0, 32767.0).astype(np.int16)


def natural_poisson_iats(n: int, mean_ms: float, rng: np.random.Generator) -> np.ndarray:
    """Natural traffic: homogeneous Poisson process (exponential IATs)."""
    return rng.exponential(mean_ms, size=n)


def natural_gaussian_iats(n: int, mean_ms: float, sigma_ms: float, rng: np.random.Generator) -> np.ndarray:
    """Natural traffic: regular-ish sender with Gaussian scheduling jitter."""
    return np.maximum(1.0, rng.normal(mean_ms, sigma_ms, size=n))


# ---------------------------------------------------------------------------
# Benchmark result containers
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkOutput:
    table: str = ""
    data: Dict[str, object] = field(default_factory=dict)
    success: bool = True
    failures: List[str] = field(default_factory=list)


def _fmt(x: float, digits: int = 2) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "n/a"
    return f"{x:.{digits}f}"


def _table(rows: List[Sequence[object]], headers: Sequence[str]) -> str:
    widths = [len(str(h)) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(str(cell)))
    head = "| " + " | ".join(str(h).ljust(widths[i]) for i, h in enumerate(headers)) + " |"
    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    body = [
        "| " + " | ".join(str(c).ljust(widths[i]) for i, c in enumerate(row)) + " |"
        for row in rows
    ]
    return "\n".join([head, sep, *body])


# ---------------------------------------------------------------------------
# Individual benchmark sections
# ---------------------------------------------------------------------------


def benchmark_audio(cfg: PhaseVeilConfig, failures: List[str]) -> Dict[str, object]:
    """Phase-coding capacity / SNR / round-trip tests on synthetic covers."""
    rows: List[Tuple[str, str]] = []
    results: Dict[str, object] = {}
    rate, cover = synth_cover()
    payload_text = b"PhaseVeil benchmark payload: covert audio phase steganography."
    for label, passphrase in (("plaintext", None), ("AES-256-GCM", "correct horse battery")):
        blob = make_blob(payload_text, passphrase, compress=True, cfg=cfg.crypto)
        stego, metrics = audio_stego.hide_payload(
            cover, rate, blob.blob, audio_cfg=cfg.audio, rs_cfg=cfg.rs, flags=blob.flags
        )
        try:
            recovered, flags, codewords = audio_stego.extract_payload(
                stego, rate, audio_cfg=cfg.audio, rs_cfg=cfg.rs
            )
            ok = parse_blob(recovered, passphrase, expect_flags=flags) == payload_text
        except Exception as exc:
            ok = False
            failures.append(f"audio round-trip ({label}): {exc}")
        rows.append(
            (
                label,
                f"{metrics.snr_db:.2f}",
                f"{metrics.psnr_db:.2f}",
                f"{metrics.payload_bytes}",
                f"{metrics.capacity_bytes}",
                f"{metrics.used_bins}",
                f"{metrics.erasures_total}",
                "PASS" if ok else "FAIL",
            )
        )
        results[label] = {
            "snr_db": metrics.snr_db,
            "psnr_db": metrics.psnr_db,
            "payload_bytes": metrics.payload_bytes,
            "capacity_bytes": metrics.capacity_bytes,
            "used_bins": metrics.used_bins,
            "erasures": metrics.erasures_total,
            "round_trip_ok": ok,
        }

    # Wrong-passphrase rejection
    blob = make_blob(b"secret", "right-pass", compress=True, cfg=cfg.crypto)
    stego, _m = audio_stego.hide_payload(cover, rate, blob.blob, audio_cfg=cfg.audio, rs_cfg=cfg.rs, flags=blob.flags)
    recovered, flags, _c = audio_stego.extract_payload(stego, rate, audio_cfg=cfg.audio, rs_cfg=cfg.rs)
    rejected = False
    try:
        parse_blob(recovered, "wrong-pass", expect_flags=flags)
    except Exception:
        rejected = True
    if not rejected:
        failures.append("wrong passphrase was accepted")
    results["wrong_passphrase_rejected"] = rejected
    rows.append(("wrong passphrase", "-", "-", "-", "-", "-", "-", "PASS" if rejected else "FAIL"))

    # Capacity sweep over cover durations
    cap_rows = []
    for seconds in (1.0, 2.0, 3.0, 5.0):
        r2, cover2 = synth_cover(seconds=seconds, seed=7 + int(seconds))
        band_mag, (k_low, k_high) = audio_stego.band_magnitudes(cover2, r2, cfg.audio)
        used, _med = audio_stego.select_bins(band_mag, k_low, k_high, cfg.audio)
        n_seg = audio_stego.segment_count(len(cover2), cfg.audio)
        cap_bits = (n_seg // cfg.audio.chunk_segments) * len(used) * cfg.audio.bits_per_bin
        cap_rows.append((f"{seconds:.0f} s", len(cover2), len(used), cap_bits // 8))
    results["capacity_sweep"] = cap_rows

    header = ("payload mode", "SNR dB", "PSNR dB", "embed B", "capacity B", "bins", "erasure bits", "round-trip")
    body = _table([list(r) for r in rows], list(header))
    sweep = _table(
        [list(r) for r in cap_rows], ["cover", "samples", "usable bins", "capacity bytes"]
    )
    return results, body + "\n\nCapacity sweep:\n" + sweep


def benchmark_timing(cfg: PhaseVeilConfig, failures: List[str]) -> Dict[str, object]:
    """Deterministic jitter recovery tests for the timing codec."""
    rng = np.random.default_rng(20240517)
    payload = b"Timing-channel jitter test payload over loopback."
    trials = 12
    recovered = 0
    for trial in range(trials):
        blob = make_blob(payload, None, compress=True, cfg=cfg.crypto)
        delays = timing_channel.encode_transmission(blob.blob, blob.flags, cfg.timing, cfg.rs)
        jitter = 3.0 + 5.0 * rng.random()  # +-3..8 ms, per trial
        iats = timing_channel.simulate_jitter(delays, rng, jitter)
        try:
            result = timing_channel.decode_transmission(iats, cfg.timing, cfg.rs)
            text = parse_blob(result.payload, None, expect_flags=result.flags)
            ok = text == payload
        except Exception as exc:
            failures.append(f"timing jitter trial {trial}: {exc}")
            ok = False
        if ok:
            recovered += 1
    # Also verify a worst-case jitter of exactly +-8 ms
    blob = make_blob(payload, None, compress=True, cfg=cfg.crypto)
    delays = timing_channel.encode_transmission(blob.blob, blob.flags, cfg.timing, cfg.rs)
    worst = timing_channel.simulate_jitter(delays, np.random.default_rng(99), 8.0)
    try:
        res = timing_channel.decode_transmission(worst, cfg.timing, cfg.rs)
        worst_ok = parse_blob(res.payload, None, expect_flags=res.flags) == payload
    except Exception as exc:
        worst_ok = False
        failures.append(f"timing worst-case jitter: {exc}")

    rows = [
        ("jitter trials (±3-8 ms)", f"{recovered}/{trials}", "PASS" if recovered == trials else "FAIL"),
        ("±8 ms worst case", "PASS" if worst_ok else "FAIL", "PASS" if worst_ok else "FAIL"),
    ]
    body = _table([list(r) for r in rows], ["timing codec test", "result", "verdict"])
    return {
        "trials": trials,
        "recovered": recovered,
        "worst_case_8ms_ok": worst_ok,
    }, body


def benchmark_detector(cfg: PhaseVeilConfig, failures: List[str]) -> Dict[str, object]:
    """Detector true-positive / false-positive accuracy on both channels."""
    rng = np.random.default_rng(4242)

    # ---- audio detector --------------------------------------------------
    audio_tp, audio_fp = 0, 0
    audio_rows = []
    for i in range(6):
        rate, cover = synth_cover(seconds=2.0 + 0.5 * i, seed=100 + i)
        blob = make_blob(b"covert payload for detector benchmark" * (i + 1), None,
                         compress=True, cfg=cfg.crypto)
        try:
            stego, _m = audio_stego.hide_payload(cover, rate, blob.blob, audio_cfg=cfg.audio,
                                                 rs_cfg=cfg.rs, flags=blob.flags)
        except Exception as exc:
            failures.append(f"audio detector embed {i}: {exc}")
            continue
        # analyse buffers directly (no temporary files needed)
        s_clean = _audio_anomaly(cover, rate, cfg)
        s_stego = _audio_anomaly(stego, rate, cfg)
        threshold = cfg.detector.audio_score_threshold
        v_clean = VERDICT_SUSPICIOUS if s_clean >= threshold else VERDICT_CLEAN
        v_stego = VERDICT_SUSPICIOUS if s_stego >= threshold else VERDICT_CLEAN
        if v_stego == VERDICT_SUSPICIOUS:
            audio_tp += 1
        else:
            failures.append(f"audio detector missed covert sample {i} (score {s_stego:.3f})")
        if v_clean == VERDICT_SUSPICIOUS:
            audio_fp += 1
            failures.append(f"audio detector false positive {i} (score {s_clean:.3f})")
        audio_rows.append((f"#{i}", f"{s_clean:.3f}", v_clean, f"{s_stego:.3f}", v_stego))

    # ---- timing detector ------------------------------------------------
    timing_tp, timing_fp_poisson, timing_fp_gauss = 0, 0, 0
    timing_rows = []
    for i in range(6):
        blob = make_blob(b"timing covert detector sample " * (i + 1), None, compress=True, cfg=cfg.crypto)
        delays = timing_channel.encode_transmission(blob.blob, blob.flags, cfg.timing, cfg.rs)
        iats = timing_channel.simulate_jitter(delays, rng, 4.0 + 3.0 * rng.random())
        rep = analyze_timing_trace(iats, cfg.detector)
        if rep.is_suspicious:
            timing_tp += 1
        else:
            failures.append(
                f"timing detector missed covert sample {i} (score {rep.anomaly_score:.3f})"
            )
        timing_rows.append((f"covert #{i}", f"{rep.anomaly_score:.3f}", rep.verdict))

    for i in range(6):
        poi = natural_poisson_iats(200, 20.0 + 5.0 * i, rng)
        rep = analyze_timing_trace(poi, cfg.detector)
        if rep.is_suspicious:
            timing_fp_poisson += 1
            failures.append(
                f"timing detector false positive on Poisson traffic {i} (score {rep.anomaly_score:.3f})"
            )
        timing_rows.append((f"poisson #{i}", f"{rep.anomaly_score:.3f}", rep.verdict))

    for i in range(6):
        gau = natural_gaussian_iats(200, 40.0, 10.0 + 3.0 * i, rng)
        rep = analyze_timing_trace(gau, cfg.detector)
        if rep.is_suspicious:
            timing_fp_gauss += 1
            failures.append(
                f"timing detector false positive on Gaussian-jitter traffic {i} "
                f"(score {rep.anomaly_score:.3f})"
            )
        timing_rows.append((f"gauss #{i}", f"{rep.anomaly_score:.3f}", rep.verdict))

    audio_body = _table(
        [list(r) for r in audio_rows],
        ["audio sample", "clean score", "clean verdict", "covert score", "covert verdict"],
    )
    timing_body = _table(
        [list(r) for r in timing_rows], ["timing sample", "anomaly score", "verdict"]
    )
    summary = _table(
        [
            ("audio true positives", f"{audio_tp}/6", "PASS" if audio_tp == 6 else "FAIL"),
            ("audio false positives", f"{audio_fp}/6", "PASS" if audio_fp == 0 else "FAIL"),
            ("timing true positives", f"{timing_tp}/6", "PASS" if timing_tp == 6 else "FAIL"),
            ("timing false positives (Poisson)", f"{timing_fp_poisson}/6",
             "PASS" if timing_fp_poisson == 0 else "FAIL"),
            ("timing false positives (Gaussian)", f"{timing_fp_gauss}/6",
             "PASS" if timing_fp_gauss == 0 else "FAIL"),
        ],
        ["detector metric", "result", "verdict"],
    )
    return {
        "audio_true_positives": audio_tp,
        "audio_false_positives": audio_fp,
        "timing_true_positives": timing_tp,
        "timing_false_positives_poisson": timing_fp_poisson,
        "timing_false_positives_gaussian": timing_fp_gauss,
    }, audio_body + "\n\n" + timing_body + "\n\n" + summary


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_benchmark(cfg: Optional[PhaseVeilConfig] = None) -> BenchmarkOutput:
    """Run every benchmark section and render the summary table."""
    cfg = cfg or PhaseVeilConfig()
    failures: List[str] = []
    out = BenchmarkOutput()

    audio_data, audio_body = benchmark_audio(cfg, failures)
    timing_data, timing_body = benchmark_timing(cfg, failures)
    detector_data, detector_body = benchmark_detector(cfg, failures)

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    out.data = {
        "config": {
            "audio": {
                "segment_length": cfg.audio.segment_length,
                "band_hz": [cfg.audio.band_low_hz, cfg.audio.band_high_hz],
                "phase_levels": cfg.audio.phase_levels,
                "chunk_segments": cfg.audio.chunk_segments,
                "bin_window": [cfg.audio.bin_mag_floor_factor, cfg.audio.bin_mag_cap_factor],
            },
            "timing": {
                "bit0_ms": cfg.timing.bit0_ms,
                "bit1_ms": cfg.timing.bit1_ms,
                "jitter_tolerance_ms": cfg.timing.jitter_tolerance_ms,
            },
            "reed_solomon": {
                "parity": cfg.rs.parity_symbols,
                "data": cfg.rs.data_symbols,
                "overhead": cfg.rs.overhead,
            },
        },
        "audio": audio_data,
        "timing": timing_data,
        "detector": detector_data,
        "peak_rss_mb": peak_kb / 1024.0,
    }

    overall = "PASS" if not failures else "FAIL"
    lines = [
        "=" * 78,
        " PhaseVeil benchmark (deterministic, synthetic fixtures only)",
        "=" * 78,
        "",
        "[1/3] Audio phase-coding: round-trip fidelity",
        audio_body,
        "",
        "[2/3] Timing channel: jitter recovery",
        timing_body,
        "",
        "[3/3] Statistical detector accuracy",
        detector_body,
        "",
        f"peak RSS: {peak_kb / 1024.0:.1f} MB   (budget 150 MB)",
        f"overall: {overall}",
    ]
    out.table = "\n".join(lines)
    out.failures = failures
    out.success = not failures
    return out