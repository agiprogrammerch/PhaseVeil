"""PhaseVeil - covert-channel, audio phase-spectrum steganography and
deterministic statistical steganalysis engine.

Submodules
----------
:mod:`phaseveil.config`          every parameter of the scheme (single source)
:mod:`phaseveil.crypto`          AES-256-GCM + zlib + PBKDF2 payload secrecy
:mod:`phaseveil.payload`         binary framing + Reed-Solomon FEC
:mod:`phaseveil.audio_stego`     frequency-domain phase-coding for 16-bit PCM wav
:mod:`phaseveil.timing_channel`  inter-packet-delay covert channel (pure + live)
:mod:`phaseveil.detector`        statistical steganalysis (zero AI/ML)
:mod:`phaseveil.benchmark`       synthetic capacity/SNR/jitter/detector benchmark
:mod:`phaseveil.cli`             ``phaseveil`` command line interface

Authorised research / educational use only: never deploy covert channels on
networks or systems you do not own or have written permission to test.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = [
    "__version__",
    "config",
    "crypto",
    "payload",
    "audio_stego",
    "timing_channel",
    "detector",
    "benchmark",
    "cli",
]