"""Tests for AES-256-GCM payload encryption and blob framing."""

from __future__ import annotations

import os

import pytest

from phaseveil.config import CryptoConfig
from phaseveil.crypto import (
    FLAG_COMPRESSED,
    FLAG_ENCRYPTED,
    CryptoError,
    PassphraseError,
    make_blob,
    parse_blob,
)

PAYLOAD = b"PhaseVeil covert payload: the quick brown fox jumps over the lazy dog."


def test_plaintext_round_trip_no_key() -> None:
    res = make_blob(PAYLOAD, None, compress=True)
    assert not res.encrypted
    assert bool(res.flags & FLAG_COMPRESSED)
    out = parse_blob(res.blob, None, expect_flags=res.flags)
    assert out == PAYLOAD


def test_encrypted_round_trip() -> None:
    res = make_blob(PAYLOAD, "correct horse battery staple", compress=True)
    assert res.encrypted
    assert bool(res.flags & FLAG_ENCRYPTED)
    out = parse_blob(res.blob, "correct horse battery staple", expect_flags=res.flags)
    assert out == PAYLOAD


def test_wrong_passphrase_rejected() -> None:
    res = make_blob(PAYLOAD, "right", compress=True)
    with pytest.raises(PassphraseError):
        parse_blob(res.blob, "wrong", expect_flags=res.flags)


def test_tampered_ciphertext_rejected() -> None:
    res = make_blob(PAYLOAD, "right", compress=True)
    blob = bytearray(res.blob)
    blob[-1] ^= 0xFF  # flip a bit inside the GCM tag
    with pytest.raises(PassphraseError):
        parse_blob(bytes(blob), "right", expect_flags=res.flags)


def test_tampered_salt_rejected() -> None:
    res = make_blob(PAYLOAD, "right", compress=True)
    blob = bytearray(res.blob)
    blob[len(b"ENC1")] ^= 0xFF
    with pytest.raises(PassphraseError):
        parse_blob(bytes(blob), "right", expect_flags=res.flags)


def test_passphrase_on_plaintext_blob_rejected() -> None:
    res = make_blob(PAYLOAD, None, compress=False)
    with pytest.raises(PassphraseError):
        parse_blob(res.blob, "some-nonempty", expect_flags=res.flags)


def test_missing_passphrase_rejected() -> None:
    res = make_blob(PAYLOAD, "right", compress=True)
    with pytest.raises(PassphraseError):
        parse_blob(res.blob, None, expect_flags=res.flags)


def test_salt_is_random_and_16_bytes() -> None:
    cfg = CryptoConfig()
    a = make_blob(b"same payload", "pw", compress=True, cfg=cfg)
    b = make_blob(b"same payload", "pw", compress=True, cfg=cfg)
    assert len(a.blob) == len(b.blob)
    assert a.blob != b.blob, "identical plaintext must not produce identical ciphertext"
    head = len(b"ENC1")
    assert a.blob[head : head + cfg.salt_bytes] != b.blob[head : head + cfg.salt_bytes]


def test_blob_layout_enc1_salt_nonce_ct() -> None:
    cfg = CryptoConfig()
    res = make_blob(b"abc", "pw", compress=True, cfg=cfg)
    assert res.blob[:4] == b"ENC1"
    expected_min = 4 + cfg.salt_bytes + cfg.nonce_bytes + cfg.tag_bits // 8
    assert len(res.blob) >= expected_min


def test_binary_payload_round_trip() -> None:
    rng = os.urandom(512)
    res = make_blob(rng, None, compress=True)
    assert parse_blob(res.blob, None, expect_flags=res.flags) == rng


def test_empty_payload_rejected_downstream_not_here() -> None:
    """Empty payloads are a framing concern; crypto must at least round-trip."""
    res = make_blob(b"", "pw", compress=True)
    assert parse_blob(res.blob, "pw", expect_flags=res.flags) == b""


def test_no_compress_flag() -> None:
    res = make_blob(b"aaaa" * 100, None, compress=False)
    assert not (res.flags & FLAG_COMPRESSED)
    assert res.blob == b"aaaa" * 100
    assert parse_blob(res.blob, None, expect_flags=res.flags) == b"aaaa" * 100


def test_compression_actually_compresses() -> None:
    text = b"compress this text " * 100
    compressed = make_blob(text, None, compress=True)
    assert len(compressed.blob) < len(text)


def test_bad_encrypted_blob_magic() -> None:
    cfg = CryptoConfig()
    res = make_blob(b"abc", "pw", compress=True, cfg=cfg)
    bad = b"XXXX" + res.blob[4:]
    with pytest.raises(CryptoError):
        parse_blob(bad, "pw", expect_flags=res.flags)


def test_truncated_encrypted_blob() -> None:
    cfg = CryptoConfig()
    res = make_blob(b"abc", "pw", compress=True, cfg=cfg)
    with pytest.raises(CryptoError):
        parse_blob(res.blob[:20], "pw", expect_flags=res.flags)