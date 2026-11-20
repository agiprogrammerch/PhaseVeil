"""Frequency-domain phase-spectrum steganography for 16-bit PCM ``.wav`` files.

Algorithm: chunked **M-PSK phase coding with inter-segment delta preservation**

Notation
--------
``x[n]``   signed 16-bit PCM cover samples, mono
``N``      FFT segment length (:data:`~phaseveil.config.AudioConfig.segment_length`)
``M``      number of segments, ``M = floor(len(x)/N)``
``X_j[k]`` ``rfft`` of segment ``j`` (``k = 0..N/2``), real input so
          ``X_j[N-k] == conj(X_j[k])`` - the mirror bins need no separate edit
``phi_j[k] = arg X_j[k]``
``B``      ordered set of *selected* mid-band bins (see bin selection below)

Embedding
---------
The cover is cut into **chunks** of ``K`` consecutive segments.  Only the
first segment of a chunk (the *carrier*) carries payload, in ``n_levels``-ary
phase shift keying: the phase of each selected bin is snapped to the
constellation point that encodes ``log2(n_levels)`` payload bits::

    bit value  v in [0, n_levels)  ->  Phi_j0[k] = 2*pi*v / n_levels

Every *following* segment of the chunk keeps the cover's inter-segment phase
difference, i.e. the delta is copied through verbatim::

    Phi_j[k] = Phi_{j-1}[k] + (phi_j[k] - phi_{j-1}[k]),   j = j0+1 .. j0+K-1

Hence ``Phi_j[k] - Phi_{j-1}[k] == phi_j[k] - phi_{j-1}[k]`` for every
follower: the *relative* phase evolution (which the ear and the spectrogram
notice) is untouched, and only the carrier segment's absolute phase is
quantised.  Magnitudes ``|X_j[k]|`` are never modified, so the spectral
envelope - and therefore the timbre - is bit-exact.

Bin selection (energy-aware)
---------------------------
The cheapest (lowest distortion) bins to re-phase are the *weak* ones: the
error energy introduced at bin ``k`` of a carrier segment is
``|X_j[k]|^2 * 4 sin^2(delta/2) / N`` (Parseval), independent of how many bits
the constellation carries.  We therefore keep the bins whose per-file median
magnitude sits between ``floor_factor`` and ``cap_factor`` times the band
median, and drop anyone inside a small *dead-band* of either boundary so that
embed and extract always agree exactly on the bin set even though 16-bit
quantisation perturbs magnitudes by ~1e-5.  The few strong harmonics (which
dominate the error energy) are excluded, which is what lifts SNR above 30 dB.

16-bit PCM quantisation safety
-----------------------------
Reconverted samples are rounded to int16 and clipped.  The per-bin phase noise
this injects is ``sigma_phi ~ (rounding_noise * sqrt(N)) / |X|``
``~ 13 / |X|`` int16 units for ``N = 2048`` - for selected bins that is
1e-3..1e-4 rad, far inside the ``+-pi/n_levels`` decision region.  Residual
risk remains where the *carrier* segment itself is near-silent (a quiet
passage): such bins are detected at extraction time from the stego file's own
magnitudes and handed to the Reed-Solomon layer as **erasures** (reedsolo
corrects up to ``nsym`` known erasures per codeword), so the payload still
decodes losslessly.

Fidelity metrics
----------------
``SNR  = 10 log10( sum x^2 / sum (x-y)^2 )``  (linear, int16 sample units)
``PSNR = 10 log10( 32767^2 / MSE )``        (full-scale reference)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
from scipy.io import wavfile

from .config import AudioConfig, ReedSolomonConfig, WIRE_VERSION
from .crypto import parse_blob
from .payload import embed_payload, parse_embedded

# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class AudioStegoError(Exception):
    """Base class for audio steganography failures."""


class CapacityError(AudioStegoError):
    """Raised when the cover cannot hold the requested payload."""


class ExtractionError(AudioStegoError):
    """Raised when no payload can be recovered from a stego file."""


# ---------------------------------------------------------------------------
# WAV I/O
# ---------------------------------------------------------------------------


def read_wav(path: str | Path) -> Tuple[int, np.ndarray]:
    """Read a mono 16-bit PCM wav; returns ``(sample_rate, int16 ndarray)``."""
    rate, data = wavfile.read(str(path))
    if data.ndim == 2:  # stereo/multi-channel -> mono average (documented)
        data = data.mean(axis=1).astype(np.int16)
    if data.dtype != np.int16:
        raise AudioStegoError(
            f"expected 16-bit PCM input, got dtype {data.dtype}; "
            "convert with: phaseveil audio convert (or re-save as 16-bit PCM)"
        )
    return int(rate), np.ascontiguousarray(data, dtype=np.int16)


def write_wav(path: str | Path, sample_rate: int, samples: np.ndarray) -> None:
    """Write mono 16-bit PCM wav."""
    if samples.dtype != np.int16:
        raise AudioStegoError("internal error: refusing to write non-int16 PCM")
    wavfile.write(str(path), int(sample_rate), np.ascontiguousarray(samples, dtype=np.int16))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StegoMetrics:
    """Quantitative fidelity report returned by :func:`hide_payload`."""

    snr_db: float
    psnr_db: float
    payload_bytes: int          # bytes actually embedded (post-FEC stream)
    payload_bits: int
    capacity_bits: int          # total available payload bits in this cover
    capacity_bytes: int
    capacity_used: float        # payload_bits / capacity_bits in [0,1]
    used_bins: int
    total_segments: int
    chunks_used: int
    bits_per_segment: int
    bitrate_bits_per_second: float
    duration_seconds: float
    erasures_total: int
    erasure_fraction: float

    def as_dict(self) -> dict:
        return {
            "snr_db": round(self.snr_db, 3),
            "psnr_db": round(self.psnr_db, 3),
            "payload_bytes": self.payload_bytes,
            "payload_bits": self.payload_bits,
            "capacity_bits": self.capacity_bits,
            "capacity_bytes": self.capacity_bytes,
            "capacity_used": round(self.capacity_used, 4),
            "used_bins": self.used_bins,
            "total_segments": self.total_segments,
            "chunks_used": self.chunks_used,
            "bits_per_segment": self.bits_per_segment,
            "bitrate_bits_per_second": round(self.bitrate_bits_per_second, 1),
            "duration_seconds": round(self.duration_seconds, 4),
            "erasures_total": self.erasures_total,
            "erasure_fraction": round(self.erasure_fraction, 4),
        }


def snr_db(cover: np.ndarray, stego: np.ndarray) -> float:
    """Signal-to-noise ratio in dB between int16 signals (linear units)."""
    a = cover.astype(np.float64)
    b = stego.astype(np.float64)
    noise = float(np.sum((a - b) ** 2))
    signal = float(np.sum(a * a))
    if noise <= 0.0:
        return float("inf")
    if signal <= 0.0:
        return float("-inf")
    return 10.0 * math.log10(signal / noise)


def psnr_db(cover: np.ndarray, stego: np.ndarray, full_scale: int = 32767) -> float:
    """Peak signal-to-noise ratio in dB (full-scale reference)."""
    a = cover.astype(np.float64)
    b = stego.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0.0:
        return float("inf")
    return 10.0 * math.log10((full_scale**2) / mse)


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def band_bins(sample_rate: int, cfg: AudioConfig) -> Tuple[int, int]:
    """Inclusive ``(k_low, k_high)`` rFFT bin indices of the embedding band."""
    return cfg.band_bins(sample_rate)


def segment_count(n_samples: int, cfg: AudioConfig) -> int:
    """Number of whole ``segment_length`` segments in ``n_samples`` samples."""
    return n_samples // cfg.segment_length


def quantization_noise_per_bin(cfg: AudioConfig) -> float:
    """RMS 16-bit rounding noise contributed to a single complex FFT bin.

    Rounding to int16 injects independent errors with variance 1/12 LSB^2; the
    DFT sums ``N`` of them, so each complex bin gets
    ``sigma = N^{-1/2}`` LSB (in amplitude units), i.e. ~13 LSB for N = 2048.
    """
    return (1.0 / math.sqrt(12.0)) * math.sqrt(cfg.segment_length)


def reliability_floor_magnitude(cfg: AudioConfig, segment_band_median: float) -> float:
    """Absolute magnitude (int16 amplitude units) below which a bin is an erasure.

    Combines the relative floor of :class:`AudioConfig` with the physical floor
    ``reliability_noise_multiple`` times the per-bin quantization noise, which is
    the point where the recovered phase stops being trustworthy.
    """
    return max(
        cfg.reliability_floor * max(segment_band_median, 1.0),
        cfg.reliability_noise_multiple * quantization_noise_per_bin(cfg),
    )


def _band_slice(k_low: int, k_high: int) -> Tuple[int, int]:
    return k_low, k_high + 1


def _iter_segments(
    samples: np.ndarray, n: int, block: int = 32
) -> Iterator[Tuple[int, np.ndarray]]:
    """Yield ``(segment_start_index, block_of_segments)`` in FFT-sized blocks."""
    total = len(samples) // n
    for s in range(0, total, block):
        e = min(s + block, total)
        yield s, samples[s * n : e * n].reshape(e - s, n).astype(np.float64)


def band_magnitudes(
    samples: np.ndarray, sample_rate: int, cfg: AudioConfig
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Per-segment band magnitudes ``[n_segments, band_width]`` (float32).

    Only the embedding band is retained, so memory stays O(segments x band)
    regardless of file length.
    """
    k_low, k_high = band_bins(sample_rate, cfg)
    n = cfg.segment_length
    blocks: List[np.ndarray] = []
    for _s, blk in _iter_segments(samples, n):
        spec = np.abs(np.fft.rfft(blk, axis=1))
        blocks.append(spec[:, k_low : k_high + 1].astype(np.float32))
    if not blocks:
        return np.zeros((0, k_high - k_low + 1), dtype=np.float32), (k_low, k_high)
    return np.concatenate(blocks, axis=0), (k_low, k_high)


def select_bins(
    band_mag: np.ndarray,
    k_low: int,
    k_high: int,
    cfg: AudioConfig,
    attempt: int = 0,
) -> Tuple[np.ndarray, float]:
    """Choose the embedding bins from per-file per-bin median magnitudes.

    Returns ``(used_bin_indices, band_median_magnitude)``.

    Determinism requirement: the *same* bins must be selected when the
    statistics are computed from the cover and from the stego file, because the
    bin order defines the bit layout.  16-bit rounding perturbs per-bin medians
    by ~0.03 %, so a bin whose median sits arbitrarily close to a selection
    boundary could otherwise change membership and silently shift every
    subsequent bit.  Two defences combine:

    * the ``bin_deadband`` drops a band around each nominal boundary, and
    * the embedder picks a ladder entry (``bin_boundary_ladder``) whose
      boundaries are demonstrably far from every median, while the extractor
      tries the same ladder until the FEC/CRC check validates the packet.
    """
    if band_mag.size == 0:
        return np.empty(0, dtype=np.int64), 0.0
    band_median = float(np.median(band_mag))
    per_bin_median = np.median(band_mag, axis=0)
    mult = cfg.bin_boundary_ladder[attempt % len(cfg.bin_boundary_ladder)]
    lo = cfg.bin_mag_floor_factor * mult * band_median
    hi = cfg.bin_mag_cap_factor * mult * band_median
    dead = cfg.bin_deadband
    keep = (per_bin_median >= lo * (1.0 + dead)) & (per_bin_median <= hi * (1.0 - dead))
    used = np.nonzero(keep)[0].astype(np.int64) + k_low
    return used, band_median


def selection_stability(
    band_mag: np.ndarray,
    k_low: int,
    k_high: int,
    cfg: AudioConfig,
    attempt: int = 0,
) -> Tuple[bool, float]:
    """Is this ladder attempt's bin choice immune to magnitude perturbations?

    Returns ``(stable, smallest_relative_boundary_distance)``.  The extractor
    recomputes the same statistic from the stego file, whose per-bin medians
    differ from the cover's by only the int16 rounding error; the choice is
    stable when every median is at least ``bin_stability_margin`` away from a
    boundary, which is an order of magnitude more than that error.
    """
    if band_mag.size == 0:
        return False, 0.0
    band_median = float(np.median(band_mag))
    per_bin_median = np.median(band_mag, axis=0)
    mult = cfg.bin_boundary_ladder[attempt % len(cfg.bin_boundary_ladder)]
    lo = cfg.bin_mag_floor_factor * mult * band_median
    hi = cfg.bin_mag_cap_factor * mult * band_median
    dead = cfg.bin_deadband
    lo_eff, hi_eff = lo * (1.0 + dead), hi * (1.0 - dead)
    # relative distance of each median to the nearest effective boundary
    d_lo = np.abs(per_bin_median - lo_eff) / max(lo_eff, 1.0)
    d_hi = np.abs(per_bin_median - hi_eff) / max(hi_eff, 1.0)
    nearest = np.minimum(d_lo, d_hi)
    smallest = float(nearest.min()) if nearest.size else 0.0
    return smallest >= cfg.bin_stability_margin, smallest


# ---------------------------------------------------------------------------
# Bit packing
# ---------------------------------------------------------------------------


def bytes_to_bits(data: bytes, order: str = "big") -> np.ndarray:
    """Pack bytes into a bit array (MSB-first by default)."""
    arr = np.frombuffer(bytes(data), dtype=np.uint8)
    bits = ((arr[:, None] >> np.arange(7, -1, -1, dtype=np.uint8)) & 1).astype(np.uint8)
    return bits.reshape(-1)


def bits_to_bytes(bits: Sequence[int]) -> bytes:
    """Inverse of :func:`bytes_to_bits` (MSB-first); drops trailing partial byte."""
    n = (len(bits) // 8) * 8
    if n == 0:
        return b""
    b = np.asarray(bits[:n], dtype=np.uint8).reshape(-1, 8)
    weights = (1 << np.arange(7, -1, -1, dtype=np.uint8)).astype(np.int32)
    values = (b.astype(np.int32) * weights).sum(axis=1).astype(np.uint8)
    return values.tobytes()


def reliability_mask_bytes(bits: Sequence[int], reliable: Sequence[bool]) -> List[bool]:
    """Map per-bit reliability to per-byte erasure flags (any bad bit => byte)."""
    out: List[bool] = []
    for i in range(0, len(bits) - 7, 8):
        out.append(not all(reliable[i : i + 8]))
    return out


# ---------------------------------------------------------------------------
# Embedding / extraction
# ---------------------------------------------------------------------------


def hide_payload(
    cover: np.ndarray,
    sample_rate: int,
    data: bytes,
    *,
    audio_cfg: AudioConfig = AudioConfig(),
    rs_cfg: ReedSolomonConfig = ReedSolomonConfig(),
    flags: int = 0,
    version: int = WIRE_VERSION,
) -> Tuple[np.ndarray, StegoMetrics]:
    """Embed ``data`` (already-encrypted blob bytes) into ``cover``.

    ``data`` is framed + Reed-Solomon protected by :mod:`phaseveil.payload`
    before phase coding, so callers pass the *raw* payload blob and the flag
    bits needed to invert the crypto layer.
    """
    stream = embed_payload(data, flags, rs_cfg, version=version)
    bits = bytes_to_bits(stream)
    n_bits = len(bits)

    cfg = audio_cfg
    n_seg = segment_count(len(cover), cfg)
    band_mag, (k_low, k_high) = band_magnitudes(cover, sample_rate, cfg)

    # Pick a ladder entry whose bin choice cannot be perturbed by 16-bit
    # rounding (see selection_stability); the extractor tries the same ladder.
    used = np.empty(0, dtype=np.int64)
    attempt = 0
    for attempt in range(len(cfg.bin_boundary_ladder)):
        candidate, _med = select_bins(band_mag, k_low, k_high, cfg, attempt)
        if candidate.size == 0:
            continue
        stable, _margin = selection_stability(band_mag, k_low, k_high, cfg, attempt)
        if stable:
            used = candidate
            break
        if used.size == 0:
            used = candidate  # remember the first non-empty fallback
    if used.size == 0:
        raise CapacityError(
            "no usable bins in the embedding band for this cover; "
            "lower bin_mag_floor_factor or use a richer audio file"
        )
    n_bins = len(used)
    bits_per_segment = n_bins * cfg.bits_per_bin
    n_chunks = n_seg // cfg.chunk_segments
    if n_chunks * bits_per_segment < n_bits:
        raise CapacityError(
            f"cover too small: needs {n_bits} bits ({len(data)} payload bytes + FEC), "
            f"have {n_chunks * bits_per_segment} ({n_chunks} chunks x {n_bins} bins "
            f"x {cfg.bits_per_bin} bits). Use a longer file or chunk_segments={cfg.chunk_segments}."
        )
    chunks_used = math.ceil(n_bits / bits_per_segment)

    stego = cover.copy()
    step = 2.0 * math.pi / cfg.phase_levels
    bits_per_bin = cfg.bits_per_bin
    K = cfg.chunk_segments
    n = cfg.segment_length
    erasures_total = 0

    for c in range(chunks_used):
        j0 = c * K
        if j0 + K > n_seg:
            break
        # --- FFT the whole chunk once (memory bounded by K*N) -------------
        spec = np.fft.rfft(
            cover[j0 * n : (j0 + K) * n].reshape(K, n).astype(np.float64), axis=1
        )
        mag = np.abs(spec)
        phase = np.angle(spec)
        new_spec = spec.copy()

        # --- carrier segment: quantise the selected bins to PSK targets ---
        carrier_bits = bits[c * bits_per_segment : (c + 1) * bits_per_segment]
        prev = phase[0, used].copy()
        for i, k in enumerate(used):
            if i * bits_per_bin + bits_per_bin <= len(carrier_bits):
                v = 0
                for b in carrier_bits[i * bits_per_bin : (i + 1) * bits_per_bin]:
                    v = (v << 1) | int(b)
                prev[i] = step * v
        new_spec[0, used] = mag[0, used] * np.exp(1j * prev)

        # --- count bins the extractor will have to treat as erasures --------
        carrier_band_median = float(np.median(mag[0, k_low : k_high + 1])) or 1.0
        floor = reliability_floor_magnitude(cfg, carrier_band_median)
        erasures_total += int(np.sum(mag[0, used] < floor)) * bits_per_bin

        # --- follower segments: preserve the cover's delta_phi -------------
        for j in range(1, K):
            delta = phase[j, used] - phase[j - 1, used]
            cur = prev + delta
            new_spec[j, used] = mag[j, used] * np.exp(1j * cur)
            prev = cur

        # --- reconstruct, round, clip (16-bit PCM quantization point) ------
        block = np.fft.irfft(new_spec, axis=1, n=n)
        segment_samples = np.clip(np.rint(block), -32768.0, 32767.0)
        stego[j0 * n : (j0 + K) * n] = segment_samples.reshape(-1)

    metrics = StegoMetrics(
        snr_db=snr_db(cover, stego) if len(cover) else float("inf"),
        psnr_db=psnr_db(cover, stego) if len(cover) else float("inf"),
        payload_bytes=len(stream),
        payload_bits=n_bits,
        capacity_bits=n_chunks * bits_per_segment,
        capacity_bytes=(n_chunks * bits_per_segment) // 8,
        capacity_used=n_bits / (n_chunks * bits_per_segment),
        used_bins=n_bins,
        total_segments=n_seg,
        chunks_used=chunks_used,
        bits_per_segment=bits_per_segment,
        bitrate_bits_per_second=n_bits / (len(cover) / sample_rate),
        duration_seconds=len(cover) / sample_rate,
        erasures_total=erasures_total,
        erasure_fraction=erasures_total / n_bits if n_bits else 0.0,
    )
    return stego, metrics


def extract_payload(
    stego: np.ndarray,
    sample_rate: int,
    *,
    audio_cfg: AudioConfig = AudioConfig(),
    rs_cfg: ReedSolomonConfig = ReedSolomonConfig(),
) -> Tuple[bytes, int, int]:
    """Recover ``(payload, flags, codewords_used)`` from a stego file.

    Extraction is key-agnostic: it returns the raw blob so the caller can
    decrypt it.  Unreliable bins are marked as Reed-Solomon erasures.
    """
    cfg = audio_cfg
    n_seg = segment_count(len(stego), cfg)
    band_mag, (k_low, k_high) = band_magnitudes(stego, sample_rate, cfg)

    # The embedder may have used any ladder entry (it picks the first *stable*
    # one), so try them all: the RS/CRC layer decides which one is right.
    last_error: Optional[Exception] = None
    for attempt in range(len(cfg.bin_boundary_ladder)):
        used, _band_median = select_bins(band_mag, k_low, k_high, cfg, attempt)
        n_bins = len(used)
        if n_bins == 0:
            continue
        bits_per_segment = n_bins * cfg.bits_per_bin
        n_chunks = n_seg // cfg.chunk_segments
        if n_chunks == 0 or bits_per_segment == 0:
            continue

        K = cfg.chunk_segments
        n = cfg.segment_length
        step = 2.0 * math.pi / cfg.phase_levels
        bits_per_bin = cfg.bits_per_bin

        bits: List[int] = []
        reliable: List[bool] = []
        for c in range(n_chunks):
            j0 = c * K
            spec = np.fft.rfft(
                stego[j0 * n : (j0 + K) * n].reshape(K, n).astype(np.float64), axis=1
            )
            mag = np.abs(spec)
            phase = np.angle(spec)
            band_med_seg = float(np.median(mag[0, k_low : k_high + 1])) or 1.0
            carrier_mag = mag[0, used]
            carrier_phase = phase[0, used]
            ok = carrier_mag >= reliability_floor_magnitude(cfg, band_med_seg)
            symbols = np.rint((carrier_phase % (2.0 * math.pi)) / step).astype(np.int64) % cfg.phase_levels
            for i in range(n_bins):
                v = int(symbols[i])
                for shift in range(bits_per_bin - 1, -1, -1):
                    bits.append((v >> shift) & 1)
                    reliable.append(bool(ok[i]))

        # Drop the trailing partial byte so the RS layer sees a whole byte count.
        n_bytes_bits = (len(bits) // 8) * 8
        bits = bits[:n_bytes_bits]
        reliable = reliable[:n_bytes_bits]
        stream = bits_to_bytes(bits)
        erasure_mask = reliability_mask_bytes(range(len(bits)), reliable)

        try:
            payload, flags, codewords = parse_embedded(stream, erasure_mask, rs_cfg)
        except Exception as exc:
            last_error = exc
            continue
        return payload, flags, codewords

    raise ExtractionError(
        f"no PhaseVeil payload recovered: {last_error or 'no usable bin selection'}"
    )


def hide_file(
    in_path: str | Path,
    out_path: str | Path,
    data: bytes,
    *,
    audio_cfg: AudioConfig = AudioConfig(),
    rs_cfg: ReedSolomonConfig = ReedSolomonConfig(),
    flags: int = 0,
) -> StegoMetrics:
    """Read ``in_path``, embed ``data``, write ``out_path``; return the metrics."""
    rate, cover = read_wav(in_path)
    stego, metrics = hide_payload(
        cover, rate, data, audio_cfg=audio_cfg, rs_cfg=rs_cfg, flags=flags
    )
    write_wav(out_path, rate, stego)
    return metrics


def extract_file(
    in_path: str | Path,
    *,
    passphrase: Optional[str] = None,
    audio_cfg: AudioConfig = AudioConfig(),
    rs_cfg: ReedSolomonConfig = ReedSolomonConfig(),
) -> bytes:
    """Extract and (if needed) decrypt a payload from a stego wav file."""
    rate, stego = read_wav(in_path)
    payload, flags, _codewords = extract_payload(
        stego, rate, audio_cfg=audio_cfg, rs_cfg=rs_cfg
    )
    return parse_blob(payload, passphrase, expect_flags=flags)
