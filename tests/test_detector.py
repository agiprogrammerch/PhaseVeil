"""Tests for the deterministic statistical steganalysis engine (zero AI/ML)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from phaseveil.config import DetectorConfig, TimingConfig
from phaseveil.crypto import make_blob
from phaseveil import audio_stego
from phaseveil.detector import (
    VERDICT_CLEAN,
    VERDICT_SUSPICIOUS,
    _audio_feature_scores,
    _kmeans2,
    _normalised_entropy,
    _timing_feature_scores,
    analyze_audio_file,
    analyze_timing_file,
    analyze_timing_trace,
    cluster_threshold,
    phase_statistics,
    timing_statistics,
)
from phaseveil.timing_channel import (
    arrival_to_iats,
    bits_to_delays,
    bytes_to_bits,
    decode_transmission,
    encode_transmission,
    simulate_jitter,
)

from conftest import (
    natural_gaussian_iats,
    natural_poisson_iats,
    synth_cover,
    write_wav,
)

COVERT_TEXT = b"covert payload embedded in the audio phase spectrum"


# ---------------------------------------------------------------------------
# Helpers / primitives
# ---------------------------------------------------------------------------


def test_normalised_entropy_extremes() -> None:
    uniform = np.full(16, 1.0)
    assert _normalised_entropy(uniform) == pytest.approx(1.0)
    single = np.zeros(16)
    single[3] = 1.0
    assert _normalised_entropy(single) == 0.0


def test_kmeans2_splits_two_modes() -> None:
    x = np.concatenate([np.full(50, 20.0), np.full(50, 60.0)])
    lo, hi, slo, shi = _kmeans2(x)
    assert lo == pytest.approx(20.0)
    assert hi == pytest.approx(60.0)
    assert slo < 1e-9 and shi < 1e-9


def test_kmeans2_unimodal_is_narrow() -> None:
    """A single Gaussian mode splits at ~1.7 sigma, never near the true gap."""
    x = np.random.default_rng(0).normal(40.0, 10.0, 400)
    lo, hi, slo, shi = _kmeans2(x)
    ratio = (hi - lo) / math.sqrt(slo**2 + shi**2)
    assert ratio < 2.0


def test_cluster_threshold_grows_with_carriers() -> None:
    det = DetectorConfig()
    few = cluster_threshold(10, det)
    many = cluster_threshold(1000, det)
    assert few < many
    assert few >= det.audio_cluster_z_base


def test_timing_feature_scores_monotone() -> None:
    det = DetectorConfig()
    rng = np.random.default_rng(0)
    covert = timing_statistics(rng.choice([20.0, 60.0], size=200))
    natural = timing_statistics(natural_poisson_iats(200, 20.0, rng))
    s_covert, sub_c = _timing_feature_scores(covert, det)
    s_clean, sub_n = _timing_feature_scores(natural, det)
    assert s_covert > s_clean
    assert 0.0 <= s_covert <= 1.0 and 0.0 <= s_clean <= 1.0


# ---------------------------------------------------------------------------
# Audio analyser
# ---------------------------------------------------------------------------


def test_clean_audio_is_clean(cfg) -> None:
    rate, cover = synth_cover(seconds=2.0, seed=11)
    metrics = phase_statistics(cover, rate, cfg.audio)
    score, _subs = _audio_feature_scores(metrics, cfg.detector, None)
    assert score < cfg.detector.audio_score_threshold
    assert metrics["phase_hist_entropy"] > 0.90


def test_covert_audio_is_suspicious(cfg) -> None:
    rate, cover = synth_cover(seconds=2.0, seed=12)
    blob = make_blob(COVERT_TEXT, None, compress=True, cfg=cfg.crypto)
    stego, _m = audio_stego.hide_payload(
        cover, rate, blob.blob, audio_cfg=cfg.audio, rs_cfg=cfg.rs, flags=blob.flags
    )
    metrics = phase_statistics(stego, rate, cfg.audio)
    score, _subs = _audio_feature_scores(metrics, cfg.detector, None)
    assert score >= cfg.detector.audio_score_threshold
    assert metrics["cluster_zmax"] > 8.0


def test_encrypted_covert_detected_same_as_plaintext(cfg) -> None:
    rate, cover = synth_cover(seconds=2.0, seed=13)
    blob = make_blob(COVERT_TEXT, "a passphrase", compress=True, cfg=cfg.crypto)
    stego, _m = audio_stego.hide_payload(
        cover, rate, blob.blob, audio_cfg=cfg.audio, rs_cfg=cfg.rs, flags=blob.flags
    )
    metrics = phase_statistics(stego, rate, cfg.audio)
    score, _subs = _audio_feature_scores(metrics, cfg.detector, None)
    assert score >= cfg.detector.audio_score_threshold


@pytest.mark.parametrize("seconds", [2.0, 3.0, 4.5])
def test_detector_across_durations(cfg, seconds) -> None:
    """Both the TP and FP rates must hold for every tested cover length."""
    rate, cover = synth_cover(seconds=seconds, seed=int(seconds * 10))
    blob = make_blob(COVERT_TEXT, None, compress=True, cfg=cfg.crypto)
    stego, _m = audio_stego.hide_payload(
        cover, rate, blob.blob, audio_cfg=cfg.audio, rs_cfg=cfg.rs, flags=blob.flags
    )
    s_clean, _ = _audio_feature_scores(
        phase_statistics(cover, rate, cfg.audio), cfg.detector, None
    )
    s_covert, _ = _audio_feature_scores(
        phase_statistics(stego, rate, cfg.audio), cfg.detector, None
    )
    assert s_clean < cfg.detector.audio_score_threshold
    assert s_covert >= cfg.detector.audio_score_threshold


def test_phase_quantised_synthetic_is_flagged(cfg) -> None:
    """A signal whose mid-band phases are exactly on the +-pi/2 grid (the
    classic two-level phase-coding marker) must be flagged."""
    rate = 44100
    x = np.zeros(rate * 2)
    t = np.arange(rate * 2) / rate
    for h in range(3, 12):
        x += (8000.0 / h) * np.sin(2 * np.pi * (300.0 * h) * t + 0.31 * h)
    samples = np.clip(np.rint(x / np.max(np.abs(x)) * 0.7 * 32767), -32768, 32767).astype(np.int16)
    metrics = phase_statistics(samples, rate, cfg.audio)
    score, _subs = _audio_feature_scores(metrics, cfg.detector, None)
    assert metrics["phase_hist_entropy"] < 1.0


def test_baseline_mode_clean_vs_clean(cfg, tmp_path) -> None:
    rate, cover = synth_cover(seconds=2.0, seed=21)
    a = tmp_path / "a.wav"
    b = tmp_path / "b.wav"
    write_wav(a, rate, cover)
    write_wav(b, rate, synth_cover(seconds=2.0, seed=22)[1])
    report = analyze_audio_file(a, baseline_path=str(b), cfg=cfg.audio, det=cfg.detector)
    assert report.verdict == VERDICT_CLEAN
    assert report.baseline_delta is not None


def test_baseline_mode_detects_covert(cfg, tmp_path) -> None:
    rate, cover = synth_cover(seconds=2.0, seed=23)
    blob = make_blob(COVERT_TEXT, None, compress=True, cfg=cfg.crypto)
    stego, _m = audio_stego.hide_payload(
        cover, rate, blob.blob, audio_cfg=cfg.audio, rs_cfg=cfg.rs, flags=blob.flags
    )
    a = tmp_path / "stego.wav"
    b = tmp_path / "cover.wav"
    write_wav(a, rate, stego)
    write_wav(b, rate, cover)
    report = analyze_audio_file(a, baseline_path=str(b), cfg=cfg.audio, det=cfg.detector)
    assert report.verdict == VERDICT_SUSPICIOUS
    assert report.anomaly_score >= cfg.detector.audio_score_threshold


def test_degenerate_in_phase_harmonic_cover_is_flagged(cfg) -> None:
    """Documented limitation, asserted as a test.

    A synthetic cover whose harmonics all start at phase ``0.4*h`` *is*
    mathematically phase-quantised: within a segment the bin phases land on a
    lattice, exactly what phase coding imposes.  No deterministic phase
    statistic can separate the two (the detector would also flag a flat
    histogram with a chi-square test).  This is why ``--baseline`` differential
    analysis exists and why natural covers are used elsewhere in the suite.
    The same harmonics with random per-harmonic phases score as clean.
    """
    rate = 44100
    x = sum(
        (7000.0 / h**1.2) * np.sin(2 * np.pi * 180 * h * np.arange(2 * rate) / rate + 0.4 * h)
        for h in range(1, 20)
    )
    in_phase = np.clip(np.rint(x / np.abs(x).max() * 0.75 * 32767), -32768, 32767).astype(np.int16)
    rng = np.random.default_rng(3)
    y = sum(
        (7000.0 / h**1.2) * np.sin(2 * np.pi * 180 * h * np.arange(2 * rate) / rate + rng.uniform(0, 6))
        for h in range(1, 20)
    )
    random_phase = np.clip(np.rint(y / np.abs(y).max() * 0.75 * 32767), -32768, 32767).astype(np.int16)
    degenerate = phase_statistics(in_phase, rate, cfg.audio)
    natural = phase_statistics(random_phase, rate, cfg.audio)
    assert degenerate["cluster_zmax"] > natural["cluster_zmax"]
    assert degenerate["phase_hist_entropy"] < natural["phase_hist_entropy"]
    score, _ = _audio_feature_scores(degenerate, cfg.detector, None)
    assert score >= cfg.detector.audio_score_threshold  # flagged: unavoidable
    nscore, _ = _audio_feature_scores(natural, cfg.detector, None)
    assert nscore < cfg.detector.audio_score_threshold  # clean control


def test_analyze_audio_file_round_trip(cfg, cover) -> None:
    rate, _samples, path = cover
    report = analyze_audio_file(str(path), cfg=cfg.audio, det=cfg.detector)
    assert report.verdict == VERDICT_CLEAN
    assert report.sample_path == str(path)
    assert report.metrics["usable_bins"] > 0


# ---------------------------------------------------------------------------
# Timing analyser
# ---------------------------------------------------------------------------


def test_covert_timing_trace_is_suspicious(cfg) -> None:
    blob = make_blob(b"timing covert payload", None, compress=True, cfg=cfg.crypto)
    delays = encode_transmission(blob.blob, blob.flags, cfg.timing, cfg.rs)
    iats = simulate_jitter(delays, np.random.default_rng(1), 5.0)
    report = analyze_timing_trace(iats, cfg.detector)
    assert report.verdict == VERDICT_SUSPICIOUS
    assert report.anomaly_score >= cfg.detector.timing_score_threshold
    assert report.metrics["bimodal_separation"] > cfg.detector.timing_bimodal_threshold


def test_poisson_traffic_is_clean(cfg) -> None:
    rng = np.random.default_rng(2)
    for i in range(6):
        iats = natural_poisson_iats(200, 20.0 + 4.0 * i, rng)
        report = analyze_timing_trace(iats, cfg.detector)
        assert report.verdict == VERDICT_CLEAN, (
            f"false positive on Poisson traffic (score {report.anomaly_score:.3f})"
        )


def test_gaussian_jitter_traffic_is_clean(cfg) -> None:
    rng = np.random.default_rng(3)
    for i in range(6):
        iats = natural_gaussian_iats(200, 40.0, 10.0 + 3.0 * i, rng)
        report = analyze_timing_trace(iats, cfg.detector)
        assert report.verdict == VERDICT_CLEAN, (
            f"false positive on Gaussian traffic (score {report.anomaly_score:.3f})"
        )


def test_regular_traffic_is_clean(cfg) -> None:
    """A perfectly regular sender (constant IAT) is degenerate but not covert."""
    iats = np.full(200, 40.0)
    report = analyze_timing_trace(iats, cfg.detector)
    assert report.verdict == VERDICT_CLEAN


def test_insufficient_samples(cfg) -> None:
    report = analyze_timing_trace(np.array([20.0, 60.0, 20.0]), cfg.detector)
    assert report.verdict == "INSUFFICIENT_DATA"
    assert report.anomaly_score == 0.0


def test_timing_statistics_values(cfg) -> None:
    rng = np.random.default_rng(4)
    delays = encode_transmission(
        make_blob(b"metric check", None, compress=True, cfg=cfg.crypto).blob,
        0x00,
        cfg.timing,
        cfg.rs,
    )
    iats = simulate_jitter(delays, rng, 4.0)
    m = timing_statistics(iats)
    assert m["n_samples"] == len(iats)
    assert m["cluster_low_ms"] == pytest.approx(20.0, abs=4.0)
    assert m["cluster_high_ms"] == pytest.approx(60.0, abs=4.0)
    assert m["mean_ms"] == pytest.approx(33.0, abs=6.0)
    # payload is compressed/whitened, so the IAT sequence is nearly
    # uncorrelated after the preamble: lag-1 autocorrelation stays small.
    assert abs(m["lag1_autocorr"]) < 0.2


def test_lag1_autocorr_detects_run_structure(cfg) -> None:
    """A structured covert stream (long runs of identical bits) leaves a strong
    positive lag-1 autocorrelation, unlike a whitened binary payload.

    Note that *compressed/encrypted* payloads are near-Bernoulli, so this
    feature only fires for structured transmissions - it is deliberately a
    secondary signal weighted below the bimodality test.
    """
    runs = np.repeat([0, 1], 40)  # 40 zeros, 40 ones, ...
    iats = bits_to_delays(np.r_[np.asarray(cfg.timing.preamble_bits), runs], cfg.timing)
    m = timing_statistics(iats)
    assert m["lag1_autocorr"] > 0.5
    assert not math.isfinite(m["bimodal_separation"]) or (
        m["bimodal_separation"] > cfg.detector.timing_bimodal_threshold
    )


def test_timing_entropy_measures_quantisation_tightness(cfg) -> None:
    """A jitter-free (or loopback-tight) two-symbol channel collapses into ~2
    ms-bins, i.e. ~1-2 bits of quantised IAT entropy; jittered symbols spread
    over more bins, so the feature is dominated by the bimodal test there."""
    delays = encode_transmission(
        make_blob(b"entropy probe", None, compress=True, cfg=cfg.crypto).blob,
        0x00,
        cfg.timing,
        cfg.rs,
    )
    tight = timing_statistics(delays)
    jittered = timing_statistics(
        simulate_jitter(delays, np.random.default_rng(7), 4.0)
    )
    natural = timing_statistics(
        natural_poisson_iats(len(delays), 33.0, np.random.default_rng(8))
    )
    assert tight["iat_entropy_bits"] < 2.0
    assert jittered["iat_entropy_bits"] > tight["iat_entropy_bits"]
    assert natural["iat_entropy_bits"] > jittered["iat_entropy_bits"]
    # the bimodal separation is the dominant discriminator in all cases
    assert tight["bimodal_separation"] > cfg.detector.timing_bimodal_threshold
    assert jittered["bimodal_separation"] > cfg.detector.timing_bimodal_threshold


def test_analyze_timing_file(cfg, tmp_path) -> None:
    from phaseveil.timing_channel import save_trace

    blob = make_blob(b"file based analysis", None, compress=True, cfg=cfg.crypto)
    delays = encode_transmission(blob.blob, blob.flags, cfg.timing, cfg.rs)
    arrival = np.cumsum(np.r_[0.0, simulate_jitter(delays, np.random.default_rng(5), 3.0)])
    trace = tmp_path / "trace.json"
    save_trace(str(trace), arrival, protocol="udp", host="127.0.0.1", port=9999)
    report = analyze_timing_file(str(trace), cfg.detector)
    assert report.verdict == VERDICT_SUSPICIOUS
    assert report.trace_path == str(trace)


def test_analyze_clean_timing_file_is_clean(cfg, tmp_path) -> None:
    from phaseveil.timing_channel import save_trace

    rng = np.random.default_rng(6)
    arrival = np.cumsum(np.r_[0.0, natural_poisson_iats(200, 20.0, rng)])
    trace = tmp_path / "clean.json"
    save_trace(str(trace), arrival, protocol="udp", host="127.0.0.1", port=9999)
    report = analyze_timing_file(str(trace), cfg.detector)
    assert report.verdict == VERDICT_CLEAN


def test_live_captured_trace_is_detectable(cfg, tmp_path) -> None:
    """A real loopback capture must be flagged (integration through the socket)."""
    import socket
    import threading
    import time

    fast = cfg.with_(
        timing=TimingConfig(
            bit0_ms=4.0, bit1_ms=12.0, jitter_tolerance_ms=2.0,
            listen_idle_timeout_ms=400.0, listen_max_seconds=30.0,
        )
    )
    blob = make_blob(b"live trace for detection", None, compress=True, cfg=fast.crypto)
    delays = encode_transmission(blob.blob, blob.flags, fast.timing, fast.rs)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    holder: dict = {}

    def _listen() -> None:
        arrival, _c = timing_channel_receive(port, fast.timing)
        holder["arrival"] = arrival

    listener = threading.Thread(target=_listen, daemon=True)
    listener.start()
    time.sleep(0.05)
    from phaseveil.timing_channel import transmit_udp

    transmit_udp("127.0.0.1", port, delays, fast.timing)
    listener.join(timeout=30.0)
    report = analyze_timing_trace(arrival_to_iats(holder["arrival"]), cfg.detector)
    assert report.verdict == VERDICT_SUSPICIOUS


def timing_channel_receive(port: int, tcfg: TimingConfig):
    from phaseveil.timing_channel import receive_udp

    return receive_udp("127.0.0.1", port, tcfg)


def test_decoded_live_payload_matches(cfg, tmp_path) -> None:
    """End-to-end: live socket -> decode -> payload equality (full stack)."""
    import socket
    import threading
    import time

    fast = cfg.with_(
        timing=TimingConfig(
            bit0_ms=4.0, bit1_ms=12.0, jitter_tolerance_ms=2.0,
            listen_idle_timeout_ms=400.0, listen_max_seconds=30.0,
        )
    )
    text = b"integrated live transmission payload"
    blob = make_blob(text, None, compress=True, cfg=fast.crypto)
    delays = encode_transmission(blob.blob, blob.flags, fast.timing, fast.rs)
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    holder: dict = {}

    def _listen() -> None:
        arrival, _c = timing_channel_receive(port, fast.timing)
        holder["arrival"] = arrival

    listener = threading.Thread(target=_listen, daemon=True)
    listener.start()
    time.sleep(0.05)
    from phaseveil.timing_channel import transmit_udp

    transmit_udp("127.0.0.1", port, delays, fast.timing)
    listener.join(timeout=30.0)
    result = decode_transmission(
        arrival_to_iats(holder["arrival"]), fast.timing, fast.rs
    )
    assert result.payload == blob.blob