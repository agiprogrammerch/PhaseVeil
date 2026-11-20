"""Authenticated payload encryption for PhaseVeil.

Layer order for ``--passphrase`` mode (this is what actually *whitens* the
payload, which both hides structure from a passive observer and destroys the
compressibility of plaintext)::

    plaintext --zlib.compress--> compressed --AES-256-GCM--> ciphertext
    blob = b"ENC1" || salt(16) || nonce(12) || ciphertext || tag(16)

The key is derived deterministically from the passphrase with
PBKDF2-HMAC-SHA256 (200k iterations) over a fresh random 16-byte salt, so the
same passphrase never yields the same keystream twice.  Because AES-GCM is an
AEAD, a wrong passphrase *or* any tampered byte fails the 128-bit
authentication tag and raises :class:`PassphraseError` - it never silently
returns garbage.

Without ``--passphrase`` the payload is only length-framed and CRC32 protected
by :mod:`phaseveil.payload`; that is the "no key" research mode and is flagged
in the frame header so extraction can pick the right path.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Optional

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes

from .config import CryptoConfig

# ---------------------------------------------------------------------------
# Flag bits stored in the frame's VERSION_FLAGS byte
# ---------------------------------------------------------------------------

#: Payload blob is AES-256-GCM encrypted (magic ``b"ENC1"``).
FLAG_ENCRYPTED: int = 0x01
#: Payload was zlib-compressed before encryption (or instead of it).
FLAG_COMPRESSED: int = 0x02
#: All flag bits currently understood by this implementation.
FLAG_KNOWN_MASK: int = FLAG_ENCRYPTED | FLAG_COMPRESSED

#: Magic prefix of an encrypted blob (encrypted-ciphertext container).
ENCRYPTED_BLOB_MAGIC: bytes = b"ENC1"


class CryptoError(Exception):
    """Raised when a payload blob cannot be authenticated or decompressed."""


class PassphraseError(CryptoError):
    """Raised when a passphrase fails or the ciphertext has been tampered."""


@dataclass(frozen=True)
class BlobResult:
    """A prepared payload blob plus the flag bits describing it."""

    blob: bytes
    flags: int
    encrypted: bool
    compressed: bool

    @property
    def plaintext_length(self) -> int:  # pragma: no cover - informational
        return len(self.blob)


def derive_key(passphrase: str, salt: bytes, cfg: CryptoConfig) -> bytes:
    """Derive a 256-bit key with PBKDF2-HMAC-SHA256 (deterministic, stateless)."""
    if not passphrase:
        raise PassphraseError("empty passphrase is not allowed")
    if len(salt) != cfg.salt_bytes:
        raise CryptoError(f"salt must be exactly {cfg.salt_bytes} bytes")
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=cfg.key_bytes,
        salt=salt,
        iterations=cfg.pbkdf2_iterations,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def make_blob(
    plaintext: bytes,
    passphrase: Optional[str],
    *,
    compress: bool = True,
    cfg: CryptoConfig = CryptoConfig(),
) -> BlobResult:
    """Wrap ``plaintext`` into an (optionally encrypted) blob.

    ``compress`` enables zlib compression, which shortens text payloads and
    removes residual redundancy before encryption.
    """
    if not isinstance(plaintext, (bytes, bytearray, memoryview)):
        raise TypeError("plaintext must be bytes-like")

    flags = 0
    body = bytes(plaintext)

    if compress:
        try:
            body = zlib.compress(body, level=6)
            flags |= FLAG_COMPRESSED
        except zlib.error as exc:  # pragma: no cover - defensive
            raise CryptoError(f"compression failed: {exc}") from exc

    if not passphrase:
        return BlobResult(blob=body, flags=flags, encrypted=False, compressed=bool(flags & FLAG_COMPRESSED))

    salt = _urandom(cfg.salt_bytes)
    nonce = _urandom(cfg.nonce_bytes)
    key = derive_key(passphrase, salt, cfg)
    ciphertext = AESGCM(key).encrypt(nonce, body, None)  # tag is appended by AESGCM
    blob = ENCRYPTED_BLOB_MAGIC + salt + nonce + ciphertext
    flags |= FLAG_ENCRYPTED
    return BlobResult(
        blob=blob, flags=flags, encrypted=True, compressed=bool(flags & FLAG_COMPRESSED)
    )


def parse_blob(
    blob: bytes,
    passphrase: Optional[str],
    *,
    expect_flags: Optional[int] = None,
    cfg: CryptoConfig = CryptoConfig(),
) -> bytes:
    """Reverse :func:`make_blob`; raise :class:`PassphraseError` on bad key/tag."""
    if not isinstance(blob, (bytes, bytearray, memoryview)):
        raise TypeError("blob must be bytes-like")
    blob = bytes(blob)

    if expect_flags is not None and not (0 <= expect_flags <= 0xFF):
        raise CryptoError("flags must be a single byte")

    encrypted = expect_flags is not None and bool(expect_flags & FLAG_ENCRYPTED)
    compressed = expect_flags is not None and bool(expect_flags & FLAG_COMPRESSED)

    if encrypted:
        if not passphrase:
            raise PassphraseError("payload is encrypted but no passphrase was supplied")
        head = len(ENCRYPTED_BLOB_MAGIC) + cfg.salt_bytes + cfg.nonce_bytes
        if len(blob) < head + cfg.tag_bits // 8:
            raise CryptoError("encrypted blob truncated")
        if blob[: len(ENCRYPTED_BLOB_MAGIC)] != ENCRYPTED_BLOB_MAGIC:
            raise CryptoError("bad encrypted-blob magic")
        off = len(ENCRYPTED_BLOB_MAGIC)
        salt = blob[off : off + cfg.salt_bytes]
        off += cfg.salt_bytes
        nonce = blob[off : off + cfg.nonce_bytes]
        off += cfg.nonce_bytes
        ciphertext = blob[off:]
        key = derive_key(passphrase, salt, cfg)
        try:
            body = AESGCM(key).decrypt(nonce, ciphertext, None)
        except InvalidTag as exc:
            # Wrong passphrase or tampered ciphertext - indistinguishable by design.
            raise PassphraseError(
                "authentication failed: wrong passphrase or tampered payload"
            ) from exc
    else:
        if passphrase:
            # Caller supplied a key for plaintext data: refuse rather than
            # silently returning something the sender never intended.
            raise PassphraseError("payload is plaintext but a passphrase was supplied")
        body = blob

    if compressed:
        try:
            body = zlib.decompress(body)
        except zlib.error as exc:
            raise CryptoError(f"decompression failed (corrupted payload?): {exc}") from exc

    return body


def _urandom(n: int) -> bytes:
    """Indirection for random draws so tests can monkey-patch determinism."""
    import os

    return os.urandom(n)