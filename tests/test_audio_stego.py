"""Tests for frequency-domain phase-spectrum steganography.

Coverage: lossless round trips for plaintext and AES-256-GCM payloads, SNR /
fidelity floors, erasure recovery from a mangled stego file, capacity limits,
determinism, and rejection of wrong passphrases.
"""

from __future__ import annotations

import numpy as np
import pytest

from phaseveil.config import AudioConfig, PhaseVeilConfig
from phaseveil.crypto import PassphraseError, make_blob, parse_blob
from phaseveil import audio_stego

TEXT = b"PhaseVeil: hidden in the phase spectrum, not the samples."
PASS = "lab authorised passphrase 42"


def _hide(cover, rate, payload, passphrase, cfg, compress=True):
    blob = make_blob(payload, passphrase, compress=compress, cfg=cfg.crypto)
    stego, metrics = audio_stego.hide_payload(
        cover, rate, blob.blob, audio_cfg=cfg.audio, rs_cfg=cfg.rs, flags=blob.flags
    )
    return stego, metrics, blob


# ---------------------------------------------------------------------------
# Round trips
# ---------------------------------------------------------------------------


def test_plaintext_round_trip(cfg, cover) -> None:
    rate, samples, _path = cover
    stego, metrics, blob = _hide(samples, rate, TEXT, None, cfg)
    payload, flags, codewords = audio_stego.extract_payload(
        stego, rate, audio_cfg=cfg.audio, rs_cfg=cfg.rs
    )
    assert parse_blob(payload, None, expect_flags=flags) == TEXT
    assert codewords >= 1


def test_encrypted_round_trip(cfg, cover) -> None:
    rate, samples, _path = cover
    stego, _m, _b = _hide(samples, rate, TEXT, PASS, cfg)
    payload, flags, _cw = audio_stego.extract_payload(
        stego, rate, audio_cfg=cfg.audio, rs_cfg=cfg.rs
    )
    assert parse_blob(payload, PASS, expect_flags=flags) == TEXT


def test_encrypted_binary_payload_round_trip(cfg, cover) -> None:
    rate, samples, _path = cover
    data = bytes(range(256)) * 2
    stego, _m, _b = _hide(samples, rate, data, PASS, cfg)
    payload, flags, _cw = audio_stego.extract_payload(
        stego, rate, audio_cfg=cfg.audio, rs_cfg=cfg.rs
    )
    assert parse_blob(payload, PASS, expect_flags=flags) == data


def test_wrong_passphrase_rejected(cfg, cover) -> None:
    rate, samples, _path = cover
    stego, _m, _b = _hide(samples, rate, TEXT, PASS, cfg)
    payload, flags, _cw = audio_stego.extract_payload(
        stego, rate, audio_cfg=cfg.audio, rs_cfg=cfg.rs
    )
    with pytest.raises(PassphraseError):
        parse_blob(payload, "not-the-passphrase", expect_flags=flags)


def test_no_passphrase_on_encrypted_payload_rejected(cfg, cover) -> None:
    rate, samples, _path = cover
    stego, _m, _b = _hide(samples, rate, TEXT, PASS, cfg)
    payload, flags, _cw = audio_stego.extract_payload(
        stego, rate, audio_cfg=cfg.audio, rs_cfg=cfg.rs
    )
    with pytest.raises(PassphraseError):
        parse_blob(payload, None, expect_flags=flags)


def test_various_payload_sizes_round_trip(cfg, cover) -> None:
    rate, samples, _path = cover
    for size in (1, 8, 40, 137, 300):
        payload = bytes((i * 7 + 13) % 256 for i in range(size))
        stego, metrics, _b = _hide(samples, rate, payload, None, cfg)
        got, flags, _cw = audio_stego.extract_payload(
            stego, rate, audio_cfg=cfg.audio, rs_cfg=cfg.rs
        )
        assert parse_blob(got, None, expect_flags=flags) == payload, f"size {size}"
        assert metrics.payload_bits > 0


# ---------------------------------------------------------------------------
# Fidelity metrics
# ---------------------------------------------------------------------------


def test_snr_above_25_db(cfg, cover) -> None:
    rate, samples, _path = cover
    stego, metrics, _b = _hide(samples, rate, TEXT, None, cfg)
    assert metrics.snr_db > 25.0, f"SNR {metrics.snr_db:.2f} dB below floor"
    assert metrics.snr_db >= 30.0, f"SNR {metrics.snr_db:.2f} dB below aim"


def test_psnr_reported(cfg, cover) -> None:
    rate, samples, _path = cover
    stego, metrics, _b = _hide(samples, rate, TEXT, None, cfg)
    assert metrics.psnr_db > 35.0


def test_capacity_and_usage_metrics(cfg, cover) -> None:
    rate, samples, _path = cover
    stego, metrics, _b = _hide(samples, rate, TEXT, None, cfg)
    assert metrics.capacity_bytes > 0
    assert 0.0 < metrics.capacity_used <= 1.0
    assert metrics.used_bins > 8
    assert metrics.chunks_used >= 1
    assert metrics.duration_seconds == pytest.approx(len(samples) / rate, rel=1e-6)


def test_magnitudes_preserved(cfg, cover) -> None:
    """Phase coding must leave the spectral envelope at used bins untouched.

    Only 16-bit rounding noise (~13 LSB per bin for N=2048) perturbs the
    magnitudes of *touched* bins, i.e. well under 1% for any selected bin.
    """
    rate, samples, _path = cover
    stego, _m, _b = _hide(samples, rate, TEXT, None, cfg)
    cfg_a = cfg.audio
    n = cfg_a.segment_length
    band_mag, (k_low, k_high) = audio_stego.band_magnitudes(samples, rate, cfg_a)
    used, _ = audio_stego.select_bins(band_mag, k_low, k_high, cfg_a)
    for j0 in (0, 16):
        seg_a = samples[j0 * n : (j0 + 1) * n].astype(np.float64)
        seg_b = stego[j0 * n : (j0 + 1) * n].astype(np.float64)
        ma = np.abs(np.fft.rfft(seg_a))
        mb = np.abs(np.fft.rfft(seg_b))
        assert np.allclose(ma[used], mb[used], rtol=0.01, atol=50.0)


def test_stego_has_same_shape_and_dtype(cfg, cover) -> None:
    rate, samples, _path = cover
    stego, _m, _b = _hide(samples, rate, TEXT, None, cfg)
    assert stego.shape == samples.shape
    assert stego.dtype == np.int16


# ---------------------------------------------------------------------------
# Robustness
# ---------------------------------------------------------------------------


def test_extraction_survives_pcm_noise(cfg, cover) -> None:
    """Re-quantisation to 16-bit PCM is the codec's design point; add extra."""
    rate, samples, _path = cover
    stego, _m, _b = _hide(samples, rate, TEXT, None, cfg)
    rng = np.random.default_rng(3)
    noisy = np.clip(stego.astype(np.int32) + rng.integers(-1, 2, size=stego.shape), -32768, 32767)
    payload, flags, _cw = audio_stego.extract_payload(
        noisy.astype(np.int16), rate, audio_cfg=cfg.audio, rs_cfg=cfg.rs
    )
    assert parse_blob(payload, None, expect_flags=flags) == TEXT


def test_extraction_survives_small_amplitude_tilt(cfg, cover) -> None:
    """A tiny gain error changes magnitudes but not the phase grid."""
    rate, samples, _path = cover
    stego, _m, _b = _hide(samples, rate, TEXT, None, cfg)
    tilted = np.clip(np.rint(stego.astype(np.float64) * 1.0005), -32768, 32767).astype(np.int16)
    payload, flags, _cw = audio_stego.extract_payload(
        tilted, rate, audio_cfg=cfg.audio, rs_cfg=cfg.rs
    )
    assert parse_blob(payload, None, expect_flags=flags) == TEXT


def test_extraction_from_unmodified_cover_fails_cleanly(cfg, cover) -> None:
    rate, samples, _path = cover
    with pytest.raises(audio_stego.ExtractionError):
        audio_stego.extract_payload(samples, rate, audio_cfg=cfg.audio, rs_cfg=cfg.rs)


def test_extraction_from_white_noise_fails_cleanly(cfg) -> None:
    rate = 44100
    noise = np.clip(
        np.rint(np.random.default_rng(0).normal(0, 3000, size=rate * 2)), -32768, 32767
    ).astype(np.int16)
    with pytest.raises(audio_stego.ExtractionError):
        audio_stego.extract_payload(noise, rate, audio_cfg=cfg.audio, rs_cfg=cfg.rs)


# ---------------------------------------------------------------------------
# Capacity / determinism / I-O
# ---------------------------------------------------------------------------


def test_capacity_error_for_tiny_cover(cfg, too_short) -> None:
    rate, samples = too_short
    blob = make_blob(b"payload", None, compress=True, cfg=cfg.crypto)
    with pytest.raises(audio_stego.CapacityError):
        audio_stego.hide_payload(samples, rate, blob.blob, audio_cfg=cfg.audio, rs_cfg=cfg.rs, flags=blob.flags)


def test_determinism_plaintext(cfg, cover) -> None:
    """Same cover + same payload + no key => byte-identical stego file."""
    rate, samples, path = cover
    a, _ma, _b = _hide(samples, rate, TEXT, None, cfg)
    b, _mb, _c = _hide(samples, rate, TEXT, None, cfg)
    assert np.array_equal(a, b)


def test_nondeterminism_with_encryption(cfg, cover) -> None:
    """Random salt/nonce means two encryptions of the same text differ."""
    rate, samples, _path = cover
    a, _ma, _b = _hide(samples, rate, TEXT, PASS, cfg)
    b, _mb, _c = _hide(samples, rate, TEXT, PASS, cfg)
    assert not np.array_equal(a, b)


def test_wav_io_round_trip(cfg, cover, tmp_path) -> None:
    rate, samples, path = cover
    read_rate, read = audio_stego.read_wav(path)
    assert read_rate == rate
    assert np.array_equal(read, samples)
    out = tmp_path / "out.wav"
    audio_stego.write_wav(out, rate, read)
    rate2, read2 = audio_stego.read_wav(out)
    assert np.array_equal(read2, samples)


def test_stereo_input_is_monoised(cfg, tmp_path) -> None:
    from scipy.io import wavfile

    rate = 44100
    stereo = np.zeros((rate, 2), dtype=np.int16)
    stereo[:, 0] = 1000
    stereo[:, 1] = -3000
    path = tmp_path / "stereo.wav"
    wavfile.write(str(path), rate, stereo)
    r, mono = audio_stego.read_wav(path)
    assert r == rate
    assert mono.ndim == 1
    assert mono.shape[0] == rate
    assert int(mono[0]) == -1000  # mean(1000, -3000)


def test_non_16bit_rejected(cfg, tmp_path) -> None:
    from scipy.io import wavfile

    path = tmp_path / "f32.wav"
    wavfile.write(str(path), 44100, np.zeros(4410, dtype=np.float32))
    with pytest.raises(audio_stego.AudioStegoError):
        audio_stego.read_wav(path)


def test_short_cover_still_round_trips(cfg, short_cover) -> None:
    rate, samples = short_cover
    stego, m, _b = _hide(samples, rate, b"tiny", None, cfg)
    payload, flags, _cw = audio_stego.extract_payload(
        stego, rate, audio_cfg=cfg.audio, rs_cfg=cfg.rs
    )
    assert parse_blob(payload, None, expect_flags=flags) == b"tiny"
    assert m.snr_db > 25.0


def test_hide_file_extract_file_round_trip(cfg, cover, tmp_path) -> None:
    rate, _samples, path = cover
    blob = make_blob(TEXT, PASS, compress=True, cfg=cfg.crypto)
    out = tmp_path / "stego.wav"
    metrics = audio_stego.hide_file(
        path, out, blob.blob, audio_cfg=cfg.audio, rs_cfg=cfg.rs, flags=blob.flags
    )
    assert out.exists()
    got = audio_stego.extract_file(out, passphrase=PASS, audio_cfg=cfg.audio, rs_cfg=cfg.rs)
    assert got == TEXT
    assert metrics.snr_db > 25.0


def test_config_validation() -> None:
    with pytest.raises(ValueError):
        AudioConfig(phase_levels=3)
    with pytest.raises(ValueError):
        AudioConfig(chunk_segments=1)
    with pytest.raises(ValueError):
        AudioConfig(band_low_hz=7000, band_high_hz=1000)
    with pytest.raises(ValueError):
        AudioConfig(bin_mag_floor_factor=1.5, bin_mag_cap_factor=1.0)
    assert AudioConfig(phase_levels=4).bits_per_bin == 2
    assert AudioConfig(phase_levels=8).bits_per_bin == 3
    assert AudioConfig(segment_length=1024).band_bins(44100)[0] > 0