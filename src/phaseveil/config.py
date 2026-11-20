"""Central, deterministic parameter set for every PhaseVeil subsystem.

All values in this module are *constants of the scheme*, chosen after empirical
validation (see ``README.md`` -> "Parameter derivation").  Nothing here depends
on random state: the same cover file, the same payload and the same parameters
must always produce byte-identical output.

Modules
-------
* :class:`AudioConfig`     -- STFT / phase-coding parameters (``audio_stego.py``)
* :class:`TimingConfig`    -- inter-packet-delay covert-channel parameters
* :class:`ReedSolomonConfig` -- Reed-Solomon FEC parameters (``payload.py``)
* :class:`CryptoConfig`    -- key-derivation parameters (``crypto.py``)
* :class:`DetectorConfig`  -- statistical steganalysis thresholds (``detector.py``)
* :class:`PhaseVeilConfig` -- aggregate config handed to callers / CLI / tests
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Tuple

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

#: Frame magic bytes 0x50 0x56 == b"PV" (``P``hase``V``eil).
MAGIC: bytes = b"\x50\x56"
#: Wire-format version written into ``VERSION_FLAGS``.
WIRE_VERSION: int = 0x01

#: Full-scale amplitude of signed 16-bit PCM.
PCM16_FULL_SCALE: int = 32767


# ---------------------------------------------------------------------------
# Audio phase-coding parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioConfig:
    """Frequency-domain phase-coding parameters.

    The cover is cut into **non-overlapping** segments of ``segment_length``
    samples.  Each segment is transformed with :func:`numpy.fft.rfft`
    (deterministic, purely analytical - no windowing is used because phase
    coding requires an *exact* unitary transform to reconstruct the signal).

    Embedding happens in **chunks** of ``chunk_segments`` consecutive
    segments.  Inside a chunk only the *first* segment carries payload, in
    ``phase_levels``-ary phase-shift-keyed (PSK/QIM) phase bins; the following
    segments keep the cover's inter-segment phase *differences* (``delta_phi``),
    which is what keeps the stego signal smooth and imperceptible.
    """

    # -- STFT ------------------------------------------------ ``segment_length``:
    #: FFT size; 2048 samples @ 44.1 kHz == 46.4 ms hop, 21.5 Hz bin width.
    segment_length: int = 2048

    # -- Embedding band (mid-frequency, avoids bass/LF rumble and HF hiss) --
    band_low_hz: float = 1000.0
    band_high_hz: float = 6000.0

    #: Number of PSK phase constellations (8 -> 3 payload bits per bin).
    phase_levels: int = 8

    #: Segments per embedding chunk (data segment + delta-preserving followers).
    chunk_segments: int = 4

    # -- Energy-aware bin selection (see README math notes) ---------------
    #: Keep bins whose per-bin median |X| >= floor_factor * band median.
    bin_mag_floor_factor: float = 0.3
    #: Keep bins whose per-bin median |X| <= cap_factor * band median.
    #: Capping removes the few strongest harmonics which dominate the
    #: embedding error energy (SNR rises by ~10 dB when they are skipped).
    bin_mag_cap_factor: float = 1.0
    #: Bins within this *relative* dead-band of a selection boundary are
    #: dropped on both sides so embed/extract usually agree on the bin set even
    #: when 16-bit PCM quantization perturbs magnitudes by ~1e-5.
    bin_deadband: float = 0.03
    #: Deterministic ladder of boundary multipliers.  The embedder picks the
    #: first entry whose boundaries sit at least ``bin_stability_margin`` away
    #: from every per-bin median (so no bin can change membership when 16-bit
    #: rounding perturbs that median), and the extractor tries the same ladder
    #: until the Reed-Solomon/CRC check validates the packet.  This removes the
    #: last cover-dependent failure mode: a bin sitting arbitrarily close to a
    #: selection boundary.
    bin_boundary_ladder: Tuple[float, ...] = (1.00, 0.99, 1.01, 0.98, 1.02, 0.96, 1.04)
    #: Minimum relative distance a per-bin median must keep from a selection
    #: boundary for the choice to be considered stable (>> the ~0.03 %
    #: perturbation int16 rounding causes).
    bin_stability_margin: float = 0.002

    #: Per-segment reliability floor: a bin is declared unreliable when its
    #: instantaneous magnitude in the *carrier* segment is below
    #: reliability_floor * that segment's band median.  An absolute floor tied
    #: to the 16-bit quantization noise scale (see
    #: :data:`reliability_noise_multiple`) is additionally enforced so that
    #: bins which are merely *weak* (but still far above the rounding noise)
    #: are trusted, while genuinely inaudible bins become RS erasures.
    reliability_floor: float = 0.01
    #: Multiple of the per-bin 16-bit quantization noise above which a bin's
    #: phase is trusted.  The rounding noise contributes ~ sigma/sqrt(N) RMS
    #: per complex bin, so this multiple bounds the phase uncertainty.
    reliability_noise_multiple: float = 20.0

    def __post_init__(self) -> None:
        if self.segment_length < 64:
            raise ValueError("segment_length must be >= 64")
        if self.phase_levels not in (2, 4, 8, 16, 32, 64):
            raise ValueError("phase_levels must be a power of two in [2, 64]")
        if self.band_low_hz >= self.band_high_hz:
            raise ValueError("band_low_hz must be < band_high_hz")
        if self.chunk_segments < 2:
            raise ValueError("chunk_segments must be >= 2 (data + preserved followers)")
        if not 0.0 < self.bin_mag_floor_factor < self.bin_mag_cap_factor:
            raise ValueError("need 0 < bin_mag_floor_factor < bin_mag_cap_factor")

    @property
    def bits_per_bin(self) -> int:
        """Payload bits carried by every selected FFT bin."""
        return int(round(math.log2(self.phase_levels)))

    def band_bins(self, sample_rate: int) -> Tuple[int, int]:
        """Return the inclusive ``(k_low, k_high)`` rFFT bin indices of the band."""
        bin_hz = sample_rate / self.segment_length
        k_low = max(1, math.ceil(self.band_low_hz / bin_hz - 1e-9))
        # A bin is fully inside the band when its centre frequency is.
        k_high = int(math.floor(self.band_high_hz / bin_hz + 1e-9))
        k_high = min(k_high, self.segment_length // 2)
        if k_high <= k_low:
            raise ValueError("embedding band contains no FFT bins; lower segment_length")
        return k_low, k_high


# ---------------------------------------------------------------------------
# Timing-channel parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TimingConfig:
    """Network inter-packet-delay (IPD) covert-channel parameters.

    ``bit0_ms`` / ``bit1_ms`` are the target inter-arrival gaps.  The 40 ms
    separation vs. an 8 ms worst-case OS scheduling jitter keeps the natural
    decision boundary (midpoint = 40 ms) >= 4x closer to the nearest symbol
    than the jitter envelope, so decoding is stable.
    """

    #: Inter-arrival gap encoding a 0 bit (ms).
    bit0_ms: float = 20.0
    #: Inter-arrival gap encoding a 1 bit (ms).
    bit1_ms: float = 60.0
    #: Alternating preamble (0xAA repeated) used for clock/bit synchronisation.
    preamble_bits: Tuple[int, ...] = field(default=(1, 0, 1, 0, 1, 0, 1, 0,
                                                   1, 0, 1, 0, 1, 0, 1, 0))
    #: Worst-case OS scheduling jitter the decoder is designed to survive (ms).
    jitter_tolerance_ms: float = 8.0
    #: Fixed benign heartbeat datagram size (bytes) - carries no covert data.
    heartbeat_bytes: int = 64
    #: Listener stops after this long without a new packet (ms).
    listen_idle_timeout_ms: float = 1500.0
    #: Hard cap on listening duration (seconds) so the CLI can never hang.
    listen_max_seconds: float = 120.0
    #: Packet send timeout for the live sender (ms).
    send_socket_timeout_ms: float = 2500.0

    def __post_init__(self) -> None:
        if self.bit0_ms <= 0 or self.bit1_ms <= self.bit0_ms:
            raise ValueError("need 0 < bit0_ms < bit1_ms")
        if len(self.preamble_bits) < 8:
            raise ValueError("preamble must be >= 8 bits for reliable sync")
        if set(self.preamble_bits) - {0, 1}:
            raise ValueError("preamble bits must be 0/1")

    @property
    def threshold_ms(self) -> float:
        """Nominal decision threshold = midpoint of the two symbols."""
        return 0.5 * (self.bit0_ms + self.bit1_ms)

    @property
    def jitter_headroom_ms(self) -> float:
        """Distance from each nominal symbol to the decision threshold."""
        return 0.5 * (self.bit1_ms - self.bit0_ms)


# ---------------------------------------------------------------------------
# Reed-Solomon / framing parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReedSolomonConfig:
    """Reed-Solomon FEC over GF(2^8) - ``reedsolo.RSCodec(nsym)``.

    A codeword is 255 bytes; ``parity_symbols`` bytes are parity, leaving
    ``data_symbols = 255 - parity_symbols`` payload bytes per codeword.  The
    framed packet is split into such blocks, so the layout costs a fixed
    ``255 / data_symbols`` overhead (~1.23x) and repairs
    ``floor(parity_symbols / 2)`` byte *errors* or up to
    ``parity_symbols`` *erasures* (unreliable bit positions, marked explicitly
    by the audio channel) per block.
    """

    #: Parity symbols per codeword.  64 repairs up to 32 byte errors *or* 64
    #: erasures per codeword, which covers even a cover whose chunks contain a
    #: fully silent carrier segment (such a segment marks ~40 bytes of
    #: erasures in the audio channel).  Default balance between channel
    #: throughput (255/(255-64) = 1.34x) and recovery power.
    parity_symbols: int = 64
    #: Total codeword length (GF(2^8) has 255 non-zero symbols).
    codeword_length: int = 255

    def __post_init__(self) -> None:
        if not 4 <= self.parity_symbols <= 128:
            raise ValueError("parity_symbols must be in [4, 128]")
        if self.parity_symbols >= self.codeword_length - 8:
            raise ValueError("too little RS data capacity left")

    @property
    def data_symbols(self) -> int:
        return self.codeword_length - self.parity_symbols

    @property
    def max_errors(self) -> int:
        """Correctable unknown-position byte errors per codeword."""
        return self.parity_symbols // 2

    @property
    def overhead(self) -> float:
        """Bandwidth inflation of the RS layer."""
        return self.codeword_length / self.data_symbols


# ---------------------------------------------------------------------------
# Crypto parameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CryptoConfig:
    """AES-256-GCM payload encryption parameters."""

    #: KDF output / AES key length in bytes (AES-256).
    key_bits: int = 256
    #: PBKDF2-HMAC-SHA256 iterations (deterministic, portable, no state).
    pbkdf2_iterations: int = 200_000
    #: Random salt length (bytes), prepended to the ciphertext blob.
    salt_bytes: int = 16
    #: AES-GCM nonce length (bytes, 96-bit IV as recommended by NIST SP 800-38D).
    nonce_bytes: int = 12
    #: GCM authentication tag length (bits).
    tag_bits: int = 128

    @property
    def key_bytes(self) -> int:
        return self.key_bits // 8

# ---------------------------------------------------------------------------
# Detector thresholds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DetectorConfig:
    """Deterministic statistical-steganalysis decision thresholds.

    Every threshold below was calibrated on synthetic clean audio /
    Poisson & Gaussian-jittered traffic versus PhaseVeil-generated covert
    samples (see ``phaseveil benchmark``).  Anomaly scores are all in
    ``[0.0, 1.0]`` and combined as fixed weighted sums, so the detector is
    fully reproducible and contains no learned / adaptive model.
    """

    #: ---- Audio phase analyser ------------------------------------------
    #: Normalised Shannon entropy of the selected-bin phase histogram above
    #: which audio looks naturally phase-random (below => suspicious).
    audio_entropy_threshold: float = 0.90
    #: Weight of the phase-entropy sub-score.
    audio_entropy_weight: float = 0.30
    #: Sub-score normalisation constant for phase entropy.
    audio_entropy_scale: float = 0.25
    #: Per-carrier grid z-score: base threshold (clean audio maxes out at ~3-4
    #: by extreme-value order) and the sub-score normalisation span (coded
    #: carriers reach ~25, so a span of 10 saturates the sub-score at 1.0).
    audio_cluster_z_base: float = 4.5
    audio_cluster_z_scale: float = 10.0
    #: Weight of the constellation z-score sub-score.
    audio_cluster_weight: float = 0.70
    #: Inter-segment (chunk-boundary) phase-jump anomaly.
    audio_boundary_weight: float = 0.0  # disabled: needs a cover reference
    #: Verdict boundary on the combined audio anomaly score.
    audio_score_threshold: float = 0.50

    #: ---- Network timing analyser ---------------------------------------
    #: Bhattacharyya-like 2-means separation ratio of the IAT clusters
    #: (natural unimodal traffic splits near ~1.2, covert traffic ~4-6).
    timing_bimodal_threshold: float = 2.0
    timing_bimodal_weight: float = 0.45
    timing_bimodal_scale: float = 2.5
    #: Normalised Shannon entropy of the ms-quantised IAT histogram; a
    #: quantised covert channel compresses into 1.0-2.5 bits of entropy while
    #: natural traffic spreads over 3.5+ bits.
    timing_entropy_threshold: float = 2.8
    timing_entropy_weight: float = 0.25
    timing_entropy_scale: float = 1.5
    #: Lag-1 autocorrelation of the IAT sequence (covert channels have strong
    #: run structure from the preamble and byte patterns; Poisson traffic ~0).
    timing_autocorr_threshold: float = 0.15
    timing_autocorr_weight: float = 0.20
    timing_autocorr_scale: float = 0.35
    #: Kolmogorov-Smirnov p-value against the better of a fitted exponential /
    #: normal marginal model (natural traffic fits; a two-point mixture does
    #: not).  Weight is deliberately low - it is the weakest single feature.
    timing_ks_weight: float = 0.10
    timing_ks_alpha: float = 0.05
    #: Verdict boundary on the combined timing anomaly score.
    timing_score_threshold: float = 0.50

    #: Minimum numbers of observations required for each verdict.
    min_audio_segments: int = 4
    min_timing_samples: int = 8


# ---------------------------------------------------------------------------
# Aggregate configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PhaseVeilConfig:
    """Bundle handed around by the library, CLI and tests."""

    audio: AudioConfig = field(default_factory=AudioConfig)
    timing: TimingConfig = field(default_factory=TimingConfig)
    rs: ReedSolomonConfig = field(default_factory=ReedSolomonConfig)
    crypto: CryptoConfig = field(default_factory=CryptoConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)

    def with_(self, **changes: object) -> "PhaseVeilConfig":
        """Return a copy with nested subsystem overrides, e.g. ``cfg.with_(audio=...)``."""
        return replace(self, **changes)


#: Process-wide default configuration.
DEFAULT_CONFIG: PhaseVeilConfig = PhaseVeilConfig()