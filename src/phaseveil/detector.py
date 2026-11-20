"""Deterministic statistical steganalysis & anomaly engine (zero AI/ML).

Two independent analysers, both built from closed-form statistics only:

Audio phase analyser
--------------------
A PhaseVeil stego file quantises the phase of ``n_levels`` selected mid-band
FFT bins, in the *carrier* segment of every chunk, onto the constellation
``2*pi*v/n_levels``.  Natural audio phases are approximately uniform on
``[0, 2pi)``.  The analyser therefore measures, over the carrier segments with
the same bin-selection geometry as the embedder:

``F1`` normalised Shannon entropy of the phase histogram (16 equal bins on
``[0, 2pi)``).  Uniform randomness is maximal entropy (~1.0); phase coding
concentrates the distribution onto ``n_levels`` points and drives the value
down to roughly ``log2(n_levels) / log2(16)``.

``F2`` constellation clustering excess.  For a tolerance ``a`` the cluster
ratio is the fraction of phases within ``a`` of a multiple of ``pi/4``; with
phases uniform on ``[0, 2pi)`` the expected ratio is exactly
``min(1, 8a/pi)``, so ``excess(a) = ratio - min(1, 8a/pi)`` is ~0 for natural
audio and approaches 1 for quantised audio.  ``F2`` is the maximum over
``a in {pi/32, pi/16, pi/8, pi/4}``, which detects any PSK order chosen by the
embedder (this strictly generalises the classic "clustering near +-pi/2"
marker of 2-level phase coding, where ``+pi/2`` is a constellation point).

An anomaly score in ``[0,1]`` is a fixed weighted sum of the normalised
deviations; the verdict is ``SUSPICIOUS`` above
``DetectorConfig.audio_score_threshold``.  Supplying ``--baseline`` replaces
the absolute references by the baseline file's own statistics, which removes
spectral-style bias for unusual covers.

Network timing analyser
-----------------------
Four purely statistical tests on inter-arrival times (IATs), all deterministic:

``T1`` 2-means separation ratio ``R = (mu_hi - mu_lo) / sqrt(sig_lo^2 + sig_hi^2)``
between the two IAT clusters (deterministic k-means initialised at the 25th/75th
percentiles).  A two-symbol covert channel yields ``R ~ 4-6``; unimodal natural
traffic (Poisson or Gaussian-jittered) splits at ``R ~ 1.1-1.5``.

``T2`` Shannon entropy (bits) of the millisecond-quantised IAT histogram.
Natural traffic spreads over many bins (~3-4 bits); a quantised covert channel
compresses into ~1-2 bits.

``T3`` lag-1 autocorrelation of the IAT sequence.  Covert streams contain long
runs (preamble, byte patterns) giving ``r1 ~ 0.3-0.9``; Poisson traffic has
``r1 ~ 0``.

``T4`` Kolmogorov-Smirnov p-value against the *better* of the maximum-likelihood
exponential and normal marginals; a two-point mixture fits neither.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import AudioConfig, DetectorConfig

#: Verdicts.
VERDICT_CLEAN = "CLEAN"
VERDICT_SUSPICIOUS = "SUSPICIOUS"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"

#: Single classification tolerance for the grid clustering test (radians).
#: pi/64 = 2.8 deg is far above the int16 quantization phase noise (~1e-3 rad)
#: yet far below the natural grid distance (pi/8 = 22.5 deg for a pi/4 grid).
CLUSTER_TOLERANCE: float = math.pi / 64
#: Constellation grid granularity used by the clustering test.
CLUSTER_GRID: float = math.pi / 4
#: Histogram resolution of the phase histogram (equal bins on [0, 2pi)).
PHASE_HIST_BINS: int = 16


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def cluster_threshold(n_carriers: int, det: Optional[DetectorConfig] = None) -> float:
    """Extreme-value-aware decision threshold for the per-carrier z-scores.

    For ``n`` roughly independent carriers the maximum of a standard normal
    sample grows like ``sqrt(2 ln n)``; adding the *base* threshold from the
    config keeps a fixed safety margin.  This bounds the clean false-alarm
    rate for arbitrarily long recordings while still sitting far below the
    ``z ~ 25`` produced by a coded carrier.
    """
    det = det or DetectorConfig()
    n = max(1, int(n_carriers))
    return det.audio_cluster_z_base + math.sqrt(2.0 * math.log(n))


def _normalised_entropy(hist: np.ndarray) -> float:
    """Shannon entropy of a histogram normalised to ``[0, 1]``."""
    h = np.asarray(hist, dtype=np.float64)
    total = h.sum()
    if total <= 0:
        return 0.0
    p = h / total
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    return float(-np.sum(p * np.log2(p)) / math.log2(len(h)))


def _kmeans2(values: np.ndarray) -> Tuple[float, float, float, float]:
    """Deterministic 2-means (init at 25th/75th percentiles).

    Returns ``(lo_mean, hi_mean, lo_std, hi_std)``.
    """
    x = np.sort(np.asarray(values, dtype=np.float64))
    lo, hi = float(np.percentile(x, 25)), float(np.percentile(x, 75))
    for _ in range(100):
        assign = x < 0.5 * (lo + hi)
        new_lo = float(x[assign].mean()) if assign.any() else lo
        new_hi = float(x[~assign].mean()) if (~assign).any() else hi
        if abs(new_lo - lo) < 1e-12 and abs(new_hi - hi) < 1e-12:
            lo, hi = new_lo, new_hi
            break
        lo, hi = new_lo, new_hi
    assign = x < 0.5 * (lo + hi)
    lo_std = float(x[assign].std()) if assign.sum() > 1 else 0.0
    hi_std = float(x[~assign].std()) if (~assign).sum() > 1 else 0.0
    return lo, hi, lo_std, hi_std


def _ks_best_natural_pvalue(x: np.ndarray) -> Tuple[float, str, float]:
    """KS p-value against the better-fitting of exponential / normal MLE models.

    ``scipy.stats`` is imported lazily: it is the single heaviest dependency
    (~50 MB RSS) and is only needed for this one timing-domain test, so audio
    workflows and the audio-only CLI paths stay light.
    Returns ``(best_p, which_model, ks_statistic)``.
    """
    from scipy import stats  # local import: keep RSS low for non-timing paths

    x = np.asarray(x, dtype=np.float64)
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if len(x) > 1 else 0.0
    if sigma <= 0 or mu <= 0:
        return 1.0, "degenerate", 0.0
    # Frozen distributions (rather than name+args) keep this stable across
    # scipy versions, which differ in how they forward ``args`` to ufuncs.
    exp_cdf = stats.expon(scale=1.0 / mu).cdf
    norm_cdf = stats.norm(loc=mu, scale=sigma).cdf
    ks_exp = stats.kstest(x, exp_cdf)
    ks_norm = stats.kstest(x, norm_cdf)
    if ks_norm.pvalue >= ks_exp.pvalue:
        return float(ks_norm.pvalue), "normal", float(ks_norm.statistic)
    return float(ks_exp.pvalue), "exponential", float(ks_exp.statistic)


# ---------------------------------------------------------------------------
# Audio analyser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioPresenceReport:
    """Structured audio steganalysis output."""

    metrics: Dict[str, float]
    anomaly_score: float
    verdict: str
    baseline_delta: Optional[Dict[str, float]] = None
    sample_path: Optional[str] = None
    baseline_path: Optional[str] = None

    @property
    def is_suspicious(self) -> bool:
        return self.verdict == VERDICT_SUSPICIOUS

    def as_dict(self) -> dict:
        return {
            "metrics": {k: round(float(v), 6) for k, v in self.metrics.items()},
            "anomaly_score": round(float(self.anomaly_score), 6),
            "verdict": self.verdict,
            "baseline_delta": (
                {k: round(float(v), 6) for k, v in self.baseline_delta.items()}
                if self.baseline_delta
                else None
            ),
        }


def phase_statistics(
    samples: np.ndarray, sample_rate: int, cfg: AudioConfig = AudioConfig()
) -> Dict[str, float]:
    """Phase-distribution statistics of one audio buffer (carrier segments).

    Two complementary statistics are computed over the selected mid-band bins of
    every *carrier* segment (i.e. the first segment of each embedding chunk):

    * ``cluster_zmax`` - for each carrier, the number of phases that fall within
      :data:`CLUSTER_TOLERANCE` of a constellation point is a Binomial
      variate under the natural (uniform-phase) null hypothesis.  Standardising
      it gives a z-score whose *maximum over carriers* is the detector's
      primary feature: natural audio maxes out at ~3-4 (extreme-value order),
      while a phase-coded carrier sits at ~25 because all its phases are
      quantised onto the grid.
    * ``phase_hist_entropy`` - pooled normalised entropy of the phase
      histogram over all carriers (secondary feature, strongest when most
      carriers are coded).
    """
    from .audio_stego import band_magnitudes, segment_count, select_bins

    n_seg = segment_count(len(samples), cfg)
    band_mag, (k_low, k_high) = band_magnitudes(samples, sample_rate, cfg)
    if band_mag.size == 0:
        raise ValueError("audio too short for analysis")
    used, _band_median = select_bins(band_mag, k_low, k_high, cfg)
    if len(used) == 0:
        raise ValueError("no usable bins in the analysis band")

    n_chunks = n_seg // cfg.chunk_segments
    if n_chunks < 1:
        raise ValueError("not enough segments for one embedding chunk")

    n = cfg.segment_length
    p0 = min(1.0, 2.0 * CLUSTER_TOLERANCE / CLUSTER_GRID)
    nb = len(used)
    z_scores: List[float] = []
    half_pi_hits = 0
    phase_total = 0
    all_phases: List[np.ndarray] = []
    for c in range(n_chunks):
        j0 = c * cfg.chunk_segments
        seg = samples[j0 * n : (j0 + 1) * n].astype(np.float64)
        ph = np.angle(np.fft.rfft(seg)[used]) % (2.0 * math.pi)
        all_phases.append(ph)
        grid = np.round(ph / CLUSTER_GRID) * CLUSTER_GRID
        dist = np.abs(ph - grid)
        hits = float(np.sum(dist <= CLUSTER_TOLERANCE))
        m0 = nb * p0
        s0 = math.sqrt(nb * p0 * (1.0 - p0))
        z_scores.append((hits - m0) / s0 if s0 > 0 else 0.0)
        # classic two-level marker: clustering at +-pi/2 (reported metric)
        d_half = np.minimum(np.abs(ph - math.pi / 2), np.abs(ph - 3 * math.pi / 2))
        half_pi_hits += int(np.sum(d_half <= math.pi / 8))
        phase_total += nb

    z = np.asarray(z_scores, dtype=np.float64)
    phases = np.concatenate(all_phases)
    hist, _ = np.histogram(phases, bins=PHASE_HIST_BINS, range=(0.0, 2.0 * math.pi))
    grid = np.round(phases / CLUSTER_GRID) * CLUSTER_GRID
    dist = np.abs(phases - grid)

    return {
        # primary: per-carrier grid z-score statistics
        "cluster_zmax": float(z.max()),
        "cluster_zmean": float(z.mean()),
        "cluster_zcount": float(np.sum(z > cluster_threshold(len(z)))),
        "cluster_ratio": float(np.mean(dist <= CLUSTER_TOLERANCE) / p0),
        # secondary: pooled phase entropy
        "phase_hist_entropy": _normalised_entropy(hist),
        # reported descriptors (not scored)
        "mean_grid_distance": float(np.mean(dist)),
        "half_pi_clustering": float(half_pi_hits / phase_total) if phase_total else 0.0,
        "usable_bins": float(nb),
        "carrier_segments": float(n_chunks),
        "phase_samples": float(phases.size),
    }


def _audio_feature_scores(
    metrics: Dict[str, float],
    det: DetectorConfig,
    baseline: Optional[Dict[str, float]] = None,
) -> Tuple[float, Dict[str, float]]:
    """Weighted anomaly sub-scores for the audio analyser.

    The primary score is the per-carrier grid z-score: clean audio tops out
    around 3-4, phase-coded carriers reach ~25, so the transformation is a
    simple clipped linear map.  Without a baseline the features are compared
    against their natural expectations (entropy ~1.0, z ~ 3); with a baseline
    the deviations from the baseline file's own statistics are used, which
    removes spectral-style bias for unusual covers.
    """
    f_ent = metrics["phase_hist_entropy"]
    f_z = metrics["cluster_zmax"]
    if baseline is not None:
        d_ent = abs(f_ent - baseline["phase_hist_entropy"])
        d_z = abs(f_z - baseline["cluster_zmax"])
        s_ent = _clamp01(d_ent / det.audio_entropy_scale)
        s_z = _clamp01(max(0.0, d_z - det.audio_cluster_z_base) / det.audio_cluster_z_scale)
    else:
        s_ent = _clamp01(max(0.0, det.audio_entropy_threshold - f_ent) / det.audio_entropy_scale)
        s_z = _clamp01(
            max(0.0, f_z - cluster_threshold(metrics["carrier_segments"], det))
            / det.audio_cluster_z_scale
        )
    total = det.audio_entropy_weight * s_ent + det.audio_cluster_weight * s_z
    return float(total), {
        "entropy_score": s_ent,
        "cluster_score": s_z,
        "cluster_threshold": cluster_threshold(metrics["carrier_segments"], det),
    }


def _clamp01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def analyze_audio_file(
    path: str | Path,
    *,
    baseline_path: Optional[str | Path] = None,
    cfg: AudioConfig = AudioConfig(),
    det: DetectorConfig = DetectorConfig(),
) -> AudioPresenceReport:
    """Analyse a ``.wav`` file for phase-coding steganography."""
    from .audio_stego import read_wav

    rate, samples = read_wav(path)
    metrics = phase_statistics(samples, rate, cfg)

    baseline_metrics = None
    if baseline_path is not None:
        brate, bsamples = read_wav(baseline_path)
        if brate != rate:
            raise ValueError(
                f"baseline sample rate {brate} != sample {rate}; cannot compare"
            )
        baseline_metrics = phase_statistics(bsamples, brate, cfg)

    score, subs = _audio_feature_scores(metrics, det, baseline_metrics)
    verdict = (
        VERDICT_SUSPICIOUS if score >= det.audio_score_threshold else VERDICT_CLEAN
    )
    out_metrics = dict(metrics)
    out_metrics.update({f"score_{k}": v for k, v in subs.items()})
    delta = None
    if baseline_metrics is not None:
        delta = {
            "phase_hist_entropy": abs(
                metrics["phase_hist_entropy"] - baseline_metrics["phase_hist_entropy"]
            ),
            "cluster_zmax": abs(metrics["cluster_zmax"] - baseline_metrics["cluster_zmax"]),
            "mean_grid_distance": abs(
                metrics["mean_grid_distance"] - baseline_metrics["mean_grid_distance"]
            ),
        }
    return AudioPresenceReport(
        metrics=out_metrics,
        anomaly_score=score,
        verdict=verdict,
        baseline_delta=delta,
        sample_path=str(path),
        baseline_path=str(baseline_path) if baseline_path else None,
    )


# ---------------------------------------------------------------------------
# Timing analyser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingPresenceReport:
    """Structured timing-channel steganalysis output."""

    metrics: Dict[str, float]
    anomaly_score: float
    verdict: str
    n_samples: int = 0
    trace_path: Optional[str] = None

    @property
    def is_suspicious(self) -> bool:
        return self.verdict == VERDICT_SUSPICIOUS

    def as_dict(self) -> dict:
        return {
            "metrics": {k: round(float(v), 6) for k, v in self.metrics.items()},
            "anomaly_score": round(float(self.anomaly_score), 6),
            "verdict": self.verdict,
            "n_samples": self.n_samples,
        }


def timing_statistics(iats_ms: Sequence[float]) -> Dict[str, float]:
    """The four statistical timing features (deterministic)."""
    x = np.asarray(iats_ms, dtype=np.float64)
    x = x[np.isfinite(x)]
    out: Dict[str, float] = {"n_samples": float(len(x))}
    if len(x) < 8:
        out.update(bimodal_separation=float("nan"), iat_entropy_bits=float("nan"),
                   lag1_autocorr=float("nan"), ks_pvalue=float("nan"))
        return out

    lo, hi, slo, shi = _kmeans2(x)
    denom = math.sqrt(slo * slo + shi * shi)
    if denom > 1e-12:
        separation = (hi - lo) / denom
    else:
        # A perfectly regular two-symbol stream (zero jitter): both clusters
        # are point masses, so the separation is unbounded, not zero.
        separation = float("inf") if hi > lo else 0.0
    out["bimodal_separation"] = float(separation)
    out["cluster_low_ms"] = lo
    out["cluster_high_ms"] = hi
    out["cluster_low_std"] = slo
    out["cluster_high_std"] = shi

    # Entropy of the ms-quantised IAT histogram (1 ms bins)
    q = np.rint(x).astype(np.int64)
    values, counts = np.unique(q, return_counts=True)
    p = counts / counts.sum()
    out["iat_entropy_bits"] = float(-np.sum(p * np.log2(p)))

    # Lag-1 autocorrelation
    xc = x - x.mean()
    var = float(np.sum(xc * xc))
    out["lag1_autocorr"] = float(np.sum(xc[:-1] * xc[1:]) / var) if var > 1e-12 else 0.0

    # KS test against the best natural marginal model
    p_best, model, ks_stat = _ks_best_natural_pvalue(x)
    out["ks_pvalue"] = float(p_best)
    out["ks_model"] = float({"normal": 0.0, "exponential": 1.0}.get(model, 2.0))
    out["ks_statistic"] = float(ks_stat)

    # Descriptives
    out["mean_ms"] = float(x.mean())
    out["std_ms"] = float(x.std(ddof=1))
    out["min_ms"] = float(x.min())
    out["max_ms"] = float(x.max())
    return out


def _timing_feature_scores(
    metrics: Dict[str, float], det: DetectorConfig
) -> Tuple[float, Dict[str, float]]:
    lo, hi = metrics["cluster_low_ms"], metrics["cluster_high_ms"]
    sep = metrics["bimodal_separation"]
    ent = metrics["iat_entropy_bits"]
    r1 = metrics["lag1_autocorr"]
    p_ks = metrics["ks_pvalue"]

    if not math.isfinite(sep):  # zero-jitter two-symbol stream
        s_bimodal = 1.0
    else:
        s_bimodal = _clamp01(
            max(0.0, sep - det.timing_bimodal_threshold) / det.timing_bimodal_scale
        )
    s_entropy = _clamp01(
        max(0.0, det.timing_entropy_threshold - ent) / det.timing_entropy_scale
    )
    s_auto = _clamp01(
        max(0.0, r1 - det.timing_autocorr_threshold) / det.timing_autocorr_scale
    )
    s_ks = _clamp01((det.timing_ks_alpha - p_ks) / det.timing_ks_alpha)
    total = (
        det.timing_bimodal_weight * s_bimodal
        + det.timing_entropy_weight * s_entropy
        + det.timing_autocorr_weight * s_auto
        + det.timing_ks_weight * s_ks
    )
    score = _clamp01(total)
    subs = {
        "bimodal_score": s_bimodal,
        "entropy_score": s_entropy,
        "autocorr_score": s_auto,
        "ks_score": s_ks,
    }
    # A bimodal split with well-separated clusters but a *single* consistent
    # gap pattern is the primary covert indicator; report it for the CLI table.
    subs["cluster_gap_ms"] = float(hi - lo)
    return score, subs


def analyze_timing_trace(
    iats_ms: Sequence[float],
    det: DetectorConfig = DetectorConfig(),
) -> TimingPresenceReport:
    """Analyse an IAT sequence for a covert timing channel."""
    metrics = timing_statistics(iats_ms)
    if metrics["n_samples"] < det.min_timing_samples:
        return TimingPresenceReport(
            metrics=metrics,
            anomaly_score=0.0,
            verdict=VERDICT_INSUFFICIENT,
            n_samples=int(metrics["n_samples"]),
        )
    score, subs = _timing_feature_scores(metrics, det)
    metrics.update(subs)
    verdict = (
        VERDICT_SUSPICIOUS if score >= det.timing_score_threshold else VERDICT_CLEAN
    )
    return TimingPresenceReport(
        metrics=metrics, anomaly_score=score, verdict=verdict,
        n_samples=int(metrics["n_samples"]),
    )


def analyze_timing_file(
    path: str | Path, det: DetectorConfig = DetectorConfig()
) -> TimingPresenceReport:
    """Analyse a saved packet-timestamp trace (see :func:`save_trace`)."""
    from .timing_channel import load_trace

    iats, meta = load_trace(path)
    report = analyze_timing_trace(iats, det)
    return TimingPresenceReport(
        metrics=report.metrics,
        anomaly_score=report.anomaly_score,
        verdict=report.verdict,
        n_samples=report.n_samples,
        trace_path=str(path),
    )