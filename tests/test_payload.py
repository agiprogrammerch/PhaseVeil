"""Tests for binary framing and the Reed-Solomon forward error correction layer."""

from __future__ import annotations

import pytest

from phaseveil.config import ReedSolomonConfig
from phaseveil.payload import (
    CRC_BYTES,
    HEADER_BYTES,
    MIN_FRAME_BYTES,
    FramingError,
    embed_payload,
    frame_packet,
    parse_embedded,
    parse_packet,
    parity_for_frame,
    rs_decode_blocks,
    rs_encode_stream,
    _decode_codeword,
)

RS = ReedSolomonConfig()


def test_frame_layout() -> None:
    frame = frame_packet(b"payload", 0x03)
    assert frame[:2] == b"\x50\x56"
    assert frame[2] == (0x01 << 4) | 0x03
    assert len(frame) == HEADER_BYTES + 7 + CRC_BYTES
    assert int.from_bytes(frame[3:7], "big") == 7


def test_frame_parse_round_trip() -> None:
    frame = frame_packet(b"PhaseVeil", 0x02)
    packet = parse_packet(frame)
    assert packet.payload == b"PhaseVeil"
    assert packet.flags == 0x02
    assert packet.version == 0x01


def test_crc_detects_corruption() -> None:
    frame = bytearray(frame_packet(b"PhaseVeil", 0x02))
    frame[10] ^= 0x01
    with pytest.raises(FramingError):
        parse_packet(bytes(frame))


def test_bad_magic_rejected() -> None:
    frame = bytearray(frame_packet(b"PhaseVeil", 0x02))
    frame[0] = 0x00
    with pytest.raises(FramingError):
        parse_packet(bytes(frame))


def test_length_mismatch_rejected() -> None:
    frame = bytearray(frame_packet(b"PhaseVeil", 0x02))
    frame[3:7] = (99).to_bytes(4, "big")  # lying header length
    with pytest.raises(FramingError):
        parse_packet(bytes(frame))


def test_truncated_frame_rejected() -> None:
    with pytest.raises(FramingError):
        parse_packet(b"PV\x12\x00\x00")
    with pytest.raises(FramingError):
        parse_packet(b"")


def test_non_zero_trailing_bytes_rejected() -> None:
    frame = frame_packet(b"payload", 0x00)
    decoded, _ = rs_decode_blocks(rs_encode_stream(frame, RS)[:255], RS)
    tampered = bytearray(decoded)
    tampered[-1] = 0x77
    with pytest.raises(FramingError):
        parse_packet(bytes(tampered))


# --------------------------------------------------------------------------
# Reed-Solomon
# --------------------------------------------------------------------------


def frame_parity(frame_len: int) -> int:
    """Parity symbols the codec uses for a frame of this length."""
    return parity_for_frame(frame_len, RS)


def test_parity_is_a_deterministic_function_of_frame_length() -> None:
    """The decoder must be able to recompute the parity from the header alone."""
    for frame_len in (20, 78, 160, 191, 200, 400, 900):
        assert frame_parity(frame_len) == RS.parity_symbols
        assert parity_for_frame(frame_len, RS) == parity_for_frame(frame_len, RS)


def test_layout_is_derivable_from_frame_length() -> None:
    """<= 191-byte frames ride in one short codeword; larger frames in blocks."""
    small = frame_packet(b"small", 0)
    assert frame_parity(len(small)) == RS.parity_symbols
    assert len(rs_encode_stream(small, RS)) == len(small) + RS.parity_symbols
    big = frame_packet(b"B" * (RS.codeword_length - RS.parity_symbols + 5), 0)
    stream = rs_encode_stream(big, RS)
    assert len(stream) % RS.codeword_length == 0
    assert len(stream) >= RS.codeword_length


def test_rs_small_mode_is_one_short_codeword() -> None:
    """A frame that fits (255 - parity) bytes uses a single short codeword."""
    frame = frame_packet(b"small payload", 0)
    stream = rs_encode_stream(frame, RS)
    assert len(stream) == len(frame) + frame_parity(len(frame))
    assert len(stream) < 255
    decoded, _ = _decode_codeword(stream, frame_parity(len(frame)), None)
    assert decoded == frame


def test_rs_large_mode_is_255_multiple() -> None:
    """Larger frames fall back to fixed 255-byte codewords."""
    frame = frame_packet(b"x" * 300, 0)
    stream = rs_encode_stream(frame, RS)
    assert len(stream) % 255 == 0
    assert len(stream) >= 2 * 255


def test_rs_small_mode_corrects_errors_up_to_capacity() -> None:
    frame = frame_packet(b"PhaseVeil RS payload " * 3, 0)
    stream = bytearray(rs_encode_stream(frame, RS))
    assert len(stream) < 255  # single-codeword mode
    for i in range(RS.max_errors):
        stream[i * 7 % len(stream)] ^= 0x5A
    decoded, repaired = _decode_codeword(bytes(stream), frame_parity(len(frame)), None)
    assert repaired >= RS.max_errors
    assert parse_packet(decoded).payload == b"PhaseVeil RS payload " * 3


def test_rs_large_mode_corrects_errors_up_to_capacity() -> None:
    frame = frame_packet(b"PhaseVeil RS payload " * 20, 0)
    stream = rs_encode_stream(frame, RS)
    assert len(stream) % 255 == 0  # multi-codeword mode
    tampered = bytearray(stream)
    for i in range(RS.max_errors):
        tampered[i * 17 % len(tampered)] ^= 0x5A
    decoded, repaired = rs_decode_blocks(bytes(tampered), RS)
    assert repaired >= RS.max_errors
    assert parse_packet(decoded).payload == b"PhaseVeil RS payload " * 20


def test_rs_corrects_erasures() -> None:
    """Erasures are the silent-carrier case: up to ``parity`` per codeword."""
    frame = frame_packet(b"erasures are cheap", 0)
    stream = rs_encode_stream(frame, RS)
    parity = frame_parity(len(frame))
    mask = [False] * len(stream)
    for i in range(0, parity, 2):
        mask[i] = True
    decoded, _ = _decode_codeword(stream, parity, [i for i, m in enumerate(mask) if m])
    assert parse_packet(decoded).payload == b"erasures are cheap"


def test_rs_gives_up_beyond_capacity() -> None:
    """More than ``parity/2`` byte errors in one codeword is unrecoverable."""
    frame = frame_packet(b"beyond repair", 0)
    stream = bytearray(rs_encode_stream(frame, RS))
    assert len(stream) < RS.codeword_length  # single-codeword mode
    for i in range(0, len(stream), 2):  # ~half the codeword corrupted
        stream[i] ^= 0xFF
    with pytest.raises(FramingError):
        _decode_codeword(bytes(stream), frame_parity(len(frame)), None)


def test_large_mode_zero_padding_tolerated() -> None:
    frame = frame_packet(b"p" * 250, 0)  # > 255 - parity -> multi-block
    stream = rs_encode_stream(frame, RS)
    assert len(stream) % 255 == 0
    decoded, _ = rs_decode_blocks(stream, RS)
    assert parse_packet(decoded).payload == b"p" * 250
    # pad beyond the frame inside the last data block must stay zero
    frame_len = len(frame)
    assert all(b == 0 for b in decoded[frame_len:])


def test_non_zero_trailing_bytes_rejected() -> None:
    frame = frame_packet(b"payload", 0)
    decoded, _ = _decode_codeword(
        rs_encode_stream(frame, RS), frame_parity(len(frame)), None
    )
    tampered = bytearray(decoded)
    tampered[-1] = 0x77
    with pytest.raises(FramingError):
        parse_packet(bytes(tampered))


def test_parse_embedded_finds_codeword_count() -> None:
    frame = frame_packet(b"single codeword payload", 0)
    stream = rs_encode_stream(frame, RS)
    payload, flags, count = parse_embedded(stream, None, RS)
    assert payload == b"single codeword payload"
    assert count == 1


def test_parse_embedded_multi_codeword() -> None:
    # force the large mode: payload larger than 255 - SMALL_NSYM
    frame = frame_packet(b"A" * (255 - RS.parity_symbols + 5), 0)
    stream = rs_encode_stream(frame, RS)
    assert len(stream) % 255 == 0
    payload, flags, count = parse_embedded(stream, None, RS)
    assert count == 2
    assert payload == b"A" * (255 - RS.parity_symbols + 5)


def test_parse_embedded_survives_corrupt_header() -> None:
    """If the systematic header is damaged the length scan must still find it."""
    frame = frame_packet(b"header-less recovery", 0)
    stream = rs_encode_stream(frame, RS)
    corrupted = b"ZZ" + stream[2:]
    payload, _flags, count = parse_embedded(corrupted, None, RS)
    assert payload == b"header-less recovery"
    assert count == 1


def test_parse_embedded_survives_corrupt_header_multiblock() -> None:
    frame = frame_packet(b"C" * 400, 0)
    stream = rs_encode_stream(frame, RS)
    corrupted = b"\x00\x00" + stream[2:]
    payload, _flags, count = parse_embedded(corrupted, None, RS)
    assert payload == b"C" * 400
    assert count >= 2


def test_parse_embedded_rejects_garbage() -> None:
    with pytest.raises(FramingError):
        parse_embedded(bytes(range(256)) * 1, None, RS)
    with pytest.raises(FramingError):
        parse_embedded(b"\x00" * 255, None, RS)


def test_parse_embedded_too_short() -> None:
    with pytest.raises(FramingError):
        parse_embedded(b"short", None, RS)


def test_parse_embedded_with_erasures() -> None:
    frame = frame_packet(b"payload with erasures " * 5, 0)
    stream = rs_encode_stream(frame, RS)
    mask = [False] * len(stream)
    for i in range(0, len(stream), 40):
        mask[i] = True
    payload, flags, count = parse_embedded(stream, mask, RS)
    assert payload == b"payload with erasures " * 5


def test_multi_codeword_with_erasures_in_each_block() -> None:
    frame = frame_packet(b"B" * 250, 0)
    stream = rs_encode_stream(frame, RS)
    assert len(stream) >= 2 * 255
    mask = [False] * len(stream)
    # deliberately place erasures in *both* codewords
    for base in (0, 255):
        for i in range(0, 20):
            mask[base + i * 11] = True
    payload, _flags, count = parse_embedded(stream, mask, RS)
    assert count == 2
    assert payload == b"B" * 250