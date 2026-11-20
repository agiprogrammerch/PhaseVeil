"""Compact binary framing + Reed-Solomon forward error correction.

Wire layout of a framed packet (all multi-byte integers are big-endian)::

    +--------+-------------+------------+-------------+---------+
    | MAGIC  | VERSION_    | PAYLOAD_   |  PAYLOAD    | CRC32   |
    | 2 B    | FLAGS 1 B   | LEN 4 B    |  PAYLOAD_LEN| 4 B     |
    | 0x5056 |             |            |  bytes      | zlib    |
    +--------+-------------+------------+-------------+---------+

``VERSION_FLAGS`` packs the wire version (high nibble, currently 0x0) and the
crypto flags from :mod:`phaseveil.crypto` (low nibble).  ``CRC32`` covers the
whole header + payload, so a mis-framed or corrupted packet is rejected
deterministically.

The framed packet is then protected by a Reed-Solomon layer
(``reedsolo.RSCodec``, GF(2^8), RS(255, k)): the frame is zero-padded and split
into ``k = 255 - nsym`` byte blocks, each independently encoded into a 255-byte
codeword.  Minor 16-bit PCM quantization damage or timing-channel jitter
therefore costs at most ``nsym/2`` repaired byte errors (or ``nsym`` known
erasures - see :mod:`phaseveil.audio_stego`, which marks unreliable bins as
erasures) per 255-byte block, and the payload still decodes losslessly.

Because the RS layer is self-delimiting through the frame header, the decoder
searches the number of codewords ``K`` from 1 upwards and accepts the first
``K`` whose decoded bytes parse as a valid frame (magic + version + CRC + a
payload length consistent with ``K``).  This removes any need to transmit an
in-band length field, which would itself be exposed to channel errors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import zlib

import reedsolo as _rs

from .config import MAGIC, ReedSolomonConfig, WIRE_VERSION

#: Bytes before the payload: 2 magic + 1 version/flags + 4 length.
HEADER_BYTES: int = 7
#: Bytes after the payload: 4-byte CRC32.
CRC_BYTES: int = 4
#: Smallest possible framed packet (empty payload).
MIN_FRAME_BYTES: int = HEADER_BYTES + CRC_BYTES


class FramingError(Exception):
    """Raised when a byte string is not a valid PhaseVeil packet."""


@dataclass(frozen=True)
class Packet:
    """A decoded framed packet."""

    payload: bytes
    flags: int
    version: int

    @property
    def encrypted(self) -> bool:
        return bool(self.flags & 0x01)

    @property
    def compressed(self) -> bool:
        return bool(self.flags & 0x02)


def frame_packet(payload: bytes, flags: int, *, version: int = WIRE_VERSION) -> bytes:
    """Build the on-wire frame for ``payload`` with crypto ``flags``."""
    if not 0 <= flags <= 0xFF:
        raise ValueError("flags must fit in one byte")
    if not 0x00 <= version <= 0x0F:
        raise ValueError("version nibble out of range")
    if len(payload) > 0xFFFFFFFF:
        raise ValueError("payload too large for the 4-byte length field")
    header = MAGIC + bytes(((version << 4) | (flags & 0x0F),)) + len(payload).to_bytes(4, "big")
    body = header + bytes(payload)
    crc = zlib.crc32(body) & 0xFFFFFFFF
    return body + crc.to_bytes(4, "big")


def parse_packet(data: bytes, *, allow_zero_padding: bool = True) -> Packet:
    """Parse and authenticate a framed packet; raise :class:`FramingError`.

    ``allow_zero_padding`` tolerates trailing zero bytes: the Reed-Solomon
    layer works on fixed 255-byte codewords, so the decoded data part is padded
    with zeros up to a block boundary.  Any *non-zero* trailing byte is still
    rejected, keeping false-positive framing essentially impossible.
    """
    data = bytes(data)
    if len(data) < MIN_FRAME_BYTES:
        raise FramingError(f"packet too short ({len(data)} < {MIN_FRAME_BYTES})")
    if data[:2] != MAGIC:
        raise FramingError("bad magic (not a PhaseVeil packet)")
    ver_flags = data[2]
    version = (ver_flags >> 4) & 0x0F
    flags = ver_flags & 0x0F
    if version != WIRE_VERSION:
        raise FramingError(f"unsupported wire version {version}")
    payload_len = int.from_bytes(data[3:7], "big")
    total = HEADER_BYTES + payload_len + CRC_BYTES
    if payload_len + MIN_FRAME_BYTES > len(data):
        raise FramingError(
            f"payload length {payload_len} exceeds packet size {len(data)}"
        )
    if len(data) > total:
        if allow_zero_padding and any(data[total:]):
            raise FramingError(
                f"unexpected {len(data) - total} trailing bytes after the frame"
            )
        if not allow_zero_padding:
            raise FramingError(f"trailing bytes: packet is {len(data)}, header says {total}")
    crc = int.from_bytes(data[total - CRC_BYTES : total], "big")
    if zlib.crc32(data[: total - CRC_BYTES]) & 0xFFFFFFFF != crc:
        raise FramingError("CRC32 mismatch (payload corrupted)")
    return Packet(payload=data[HEADER_BYTES : HEADER_BYTES + payload_len], flags=flags, version=version)


# ---------------------------------------------------------------------------
# Reed-Solomon layer
# ---------------------------------------------------------------------------


def _rsc(nsym: int, nsize: int) -> "_rs.RSCodec":
    return _rs.RSCodec(nsym, nsize)


def parity_for_frame(frame_len: int, rs_cfg: ReedSolomonConfig) -> int:
    """Deterministic parity-symbol count for a frame of ``frame_len`` bytes.

    The parity is the constant configuration value; the decoder derives the
    *codeword layout* (one short codeword vs. ``ceil(frame/k)`` full codewords)
    from the frame length it reads out of the intact systematic header, so no
    out-of-band parity field is ever needed.
    """
    if frame_len <= 0:
        raise ValueError("frame length must be positive")
    return int(rs_cfg.parity_symbols)


def rs_encode_stream(data: bytes, rs_cfg: ReedSolomonConfig) -> bytes:
    """RS-encode ``data`` (a framed packet) into the embedded byte stream.

    Two systematic modes, chosen by size so the decoder can always recover:

    * **small payload** - one codeword of exactly
      ``len(data) + parity`` bytes (fits when ``len(data) <= 255 - parity``).
      A 60-byte frame becomes a ~124-byte codeword instead of a 255-byte
      block, which is what keeps short timing transmissions fast.
    * **large payload** - the frame is zero-padded and split into
      ``k = 255 - parity`` blocks, each encoded into a 255-byte codeword
      (required once the frame exceeds ``255 - parity`` bytes, because a
      GF(2^8) codeword is at most 255 bytes long).
    """
    if len(data) == 0:
        raise ValueError("nothing to encode")
    parity = rs_cfg.parity_symbols
    if len(data) + parity <= rs_cfg.codeword_length:
        return bytes(_rsc(parity, len(data) + parity).encode(bytearray(data)))
    padded = bytes(data)
    pad = (-len(padded)) % rs_cfg.data_symbols
    padded += b"\x00" * pad
    rsc = _rsc(rs_cfg.parity_symbols, rs_cfg.codeword_length)
    out = bytearray()
    for i in range(0, len(padded), rs_cfg.data_symbols):
        out += rsc.encode(bytearray(padded[i : i + rs_cfg.data_symbols]))
    return bytes(out)


def _decode_codeword(
    codeword: bytes,
    nsym: int,
    erasures: Optional[List[int]],
) -> Tuple[bytes, int]:
    """Decode one codeword of length ``len(codeword)`` with ``nsym`` parity."""
    rsc = _rsc(nsym, len(codeword))
    try:
        if erasures:
            dec, _full, errata = rsc.decode(bytearray(codeword), erase_pos=erasures)
        else:
            dec, _full, errata = rsc.decode(bytearray(codeword))
    except Exception as exc:  # reedsolo raises ReedSolomonError / ValueError
        raise FramingError(f"RS decode failed ({nsym} parity): {exc}") from exc
    return bytes(dec), len(errata)


def rs_decode_blocks(
    stream: bytes,
    rs_cfg: ReedSolomonConfig,
    erasure_mask: Optional[Sequence[bool]] = None,
) -> Tuple[bytes, int]:
    """RS-decode a stream of **255-byte** codewords (large-payload mode).

    Returns ``(decoded_k_bytes, repaired_byte_count)``.
    Raises :class:`FramingError` if a codeword is beyond repair.
    """
    k = rs_cfg.data_symbols
    if len(stream) % rs_cfg.codeword_length:
        raise FramingError(
            f"stream length {len(stream)} is not a multiple of the RS codeword size"
        )
    out = bytearray()
    repaired = 0
    for bstart in range(0, len(stream), rs_cfg.codeword_length):
        block = stream[bstart : bstart + rs_cfg.codeword_length]
        erasures = _erasures(erasure_mask, bstart, rs_cfg.codeword_length)
        dec, n_err = _decode_codeword(block, rs_cfg.parity_symbols, erasures)
        if len(dec) != k:
            raise FramingError(f"RS block decoded to {len(dec)} bytes, expected {k}")
        repaired += n_err
        out += dec
    return bytes(out), repaired


def _erasures(
    erasure_mask: Optional[Sequence[bool]], base: int, length: int
) -> Optional[List[int]]:
    """Erasure positions (codeword-relative) for a slice of a byte mask."""
    if erasure_mask is None:
        return None
    return [
        i
        for i in range(length)
        if base + i < len(erasure_mask) and erasure_mask[base + i]
    ]


def embed_payload(
    payload: bytes, flags: int, rs_cfg: ReedSolomonConfig, *, version: int = WIRE_VERSION
) -> bytes:
    """Frame ``payload`` and RS-protect it -> bytes handed to a channel."""
    frame = frame_packet(payload, flags, version=version)
    return rs_encode_stream(frame, rs_cfg)


def parse_embedded(
    stream: bytes,
    erasure_mask: Optional[Sequence[bool]] = None,
    rs_cfg: ReedSolomonConfig = ReedSolomonConfig(),
) -> Tuple[bytes, int, int]:
    """Recover ``(payload, flags, codewords_used)`` from a decoded channel stream.

    Because RS codewords are *systematic*, an intact frame header is directly
    readable at offset 0, which gives the frame length and therefore tells us
    exactly how many parity symbols and bytes to decode.  If the header itself
    is damaged, the decoder falls back to scanning the single-codeword length
    and the 255-byte block count, validating every candidate against
    magic + version + CRC + length consistency, so a false positive is
    effectively impossible.

    Returns the number of *codewords* consumed (1 for the small-payload mode).
    """
    total = len(stream)
    if total < rs_cfg.parity_symbols + MIN_FRAME_BYTES:
        raise FramingError(
            f"channel stream too short: {total} bytes cannot hold one RS codeword"
        )

    # ---- fast path: intact systematic header ---------------------------
    if stream[:2] == MAGIC:
        try:
            frame_len = HEADER_BYTES + int.from_bytes(stream[3:7], "big") + CRC_BYTES
        except Exception:
            frame_len = -1
        if MIN_FRAME_BYTES <= frame_len <= total:
            parity = parity_for_frame(frame_len, rs_cfg)
            if frame_len + parity <= total:
                cand = stream[: frame_len + parity]
                mask = _erasures(erasure_mask, 0, len(cand))
                try:
                    decoded, _rep = _decode_codeword(cand, parity, mask)
                    packet = parse_packet(decoded)
                    if HEADER_BYTES + len(packet.payload) + CRC_BYTES == frame_len:
                        return packet.payload, packet.flags, 1
                except FramingError:
                    pass
            else:
                blocks_needed = -(-frame_len // rs_cfg.data_symbols)
                cut = min(total, blocks_needed * rs_cfg.codeword_length)
                try:
                    decoded, _rep = rs_decode_blocks(
                        stream[:cut], rs_cfg, erasure_mask
                    )
                    packet = parse_packet(decoded)
                    if HEADER_BYTES + len(packet.payload) + CRC_BYTES == frame_len:
                        return packet.payload, packet.flags, blocks_needed
                except FramingError:
                    pass

    # ---- fallback 1: single-codeword length scan -----------------------
    limit = min(total, rs_cfg.codeword_length)
    for length in range(rs_cfg.parity_symbols + MIN_FRAME_BYTES, limit + 1):
        parity = min(rs_cfg.parity_symbols, rs_cfg.codeword_length - length)
        if parity < 8:
            continue
        mask = _erasures(erasure_mask, 0, length)
        try:
            decoded, _rep = _decode_codeword(stream[:length], parity, mask)
            packet = parse_packet(decoded)
        except FramingError:
            continue
        if HEADER_BYTES + len(packet.payload) + CRC_BYTES <= length:
            return packet.payload, packet.flags, 1

    # ---- fallback 2: 255-byte block count scan -------------------------
    for cnt in range(1, total // rs_cfg.codeword_length + 1):
        cut = cnt * rs_cfg.codeword_length
        mask = _erasures(erasure_mask, 0, cut)
        try:
            decoded, _rep = rs_decode_blocks(stream[:cut], rs_cfg, mask)
            packet = parse_packet(decoded)
        except FramingError:
            continue
        frame_len = HEADER_BYTES + len(packet.payload) + CRC_BYTES
        if (cnt - 1) * rs_cfg.data_symbols < frame_len <= cnt * rs_cfg.data_symbols:
            return packet.payload, packet.flags, cnt

    raise FramingError("no valid PhaseVeil packet found in the channel stream")