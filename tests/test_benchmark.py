"""Tests for the synthetic benchmark harness itself (structure, determinism, RAM)."""

from __future__ import annotations

import resource

import numpy as np
import pytest

from phaseveil.benchmark import (
    BenchmarkOutput,
    benchmark_audio,
    benchmark_detector,
    benchmark_timing,
    natural_gaussian_iats,
    natural_poisson_iats,
    run_benchmark,
    synth_cover,
)
from phaseveil.config import PhaseVeilConfig


def test_synth_cover_is_16bit_and_bounded() -> None:
    rate, samples = synth_cover(seconds=1.0)
    assert rate == 44100
    assert samples.dtype == np.int16
    assert samples.ndim == 1
    assert samples.size == rate
    assert int(np.abs(samples).max()) <= 32767
    assert float(np.abs(samples).max()) > 1000  # not silent


def test_synth_cover_deterministic() -> None:
    _, a = synth_cover(seconds=0.8, seed=5)
    _, b = synth_cover(seconds=0.8, seed=5)
    assert np.array_equal(a, b)


def test_natural_traffic_generators() -> None:
    rng = np.random.default_rng(0)
    poi = natural_poisson_iats(500, 20.0, rng)
    gau = natural_gaussian_iats(500, 40.0, 10.0, rng)
    assert poi.shape == (500,) and gau.shape == (500,)
    assert poi.min() > 0 and gau.min() > 0
    # exponential mean ~ mean_ms; gaussian mean ~ mean_ms
    assert poi.mean() == pytest.approx(20.0, rel=0.2)
    assert gau.mean() == pytest.approx(40.0, abs=3.0)


def test_benchmark_sections_pass() -> None:
    cfg = PhaseVeilConfig()
    failures: list = []
    a_data, _a = benchmark_audio(cfg, failures)
    t_data, _t = benchmark_timing(cfg, failures)
    d_data, _d = benchmark_detector(cfg, failures)
    assert a_data["plaintext"]["round_trip_ok"] is True
    assert a_data["AES-256-GCM"]["round_trip_ok"] is True
    assert a_data["wrong_passphrase_rejected"] is True
    assert t_data["recovered"] == t_data["trials"]
    assert t_data["worst_case_8ms_ok"] is True
    assert d_data["audio_true_positives"] == 6
    assert d_data["audio_false_positives"] == 0
    assert d_data["timing_true_positives"] == 6
    assert d_data["timing_false_positives_poisson"] == 0
    assert d_data["timing_false_positives_gaussian"] == 0
    assert not failures, failures


def test_benchmark_full_run_is_deterministic() -> None:
    """Plaintext/audio-free-of-randomness sections reproduce exactly.

    The AES-256-GCM row intentionally uses a fresh random salt+nonce on every
    run (that is the property that stops identical plaintexts from producing
    identical ciphertext), so its SNR varies in the last decimal digits.
    """
    first = run_benchmark()
    second = run_benchmark()
    assert first.success and second.success
    for key in ("timing", "detector"):
        assert first.data[key] == second.data[key], key
    assert first.data["audio"]["plaintext"] == second.data["audio"]["plaintext"]
    assert first.data["audio"]["AES-256-GCM"]["round_trip_ok"] is True
    assert second.data["audio"]["AES-256-GCM"]["round_trip_ok"] is True
    # the encrypted SNR only moves within a fraction of a dB
    assert abs(
        first.data["audio"]["AES-256-GCM"]["snr_db"]
        - second.data["audio"]["AES-256-GCM"]["snr_db"]
    ) < 1.0


def test_benchmark_reports_peak_rss() -> None:
    out = run_benchmark()
    assert out.data["peak_rss_mb"] > 0
    assert "peak RSS" in out.table
    assert "overall: PASS" in out.table


def test_benchmark_output_defaults() -> None:
    default = BenchmarkOutput()
    assert default.table == ""
    assert default.success is True
    assert default.failures == []


def test_rss_stays_under_budget_after_run() -> None:
    """The documented 150 MB budget must hold after a full benchmark run."""
    run_benchmark()
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    assert peak_mb < 150.0, f"peak RSS {peak_mb:.1f} MB exceeds budget"