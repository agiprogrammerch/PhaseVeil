"""Shared pytest fixtures.

All audio fixtures are generated **synthetically in memory** (no binary media
is committed to the repository), every random draw is seeded, and every
network test binds to an ephemeral port on ``127.0.0.1``.
"""

from __future__ import annotations

import math
import socket
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from phaseveil.config import PhaseVeilConfig

# ---------------------------------------------------------------------------
# Synthetic audio generation
# ---------------------------------------------------------------------------


def synth_cover(
    seconds: float = 3.0,
    sample_rate: int = 44100,
    *,
    seed: int = 1234,
    fundamental: float = 220.0,
    harmonics: int = 24,
    noise_sigma: float = 120.0,
    quiet_patch: bool = True,
) -> Tuple[int, np.ndarray]:
    """Deterministic multi-harmonic 16-bit PCM cover.

    Tapering harmonics with per-harmonic amplitude modulation, a faint noise
    floor, and a near-silent passage so erasure handling is exercised.  The
    harmonics are deliberately NOT bin-aligned so natural phases stay spread
    over [0, 2pi) (a fully in-phase, bin-aligned signal would be a degenerate
    cover, discussed in README.md limitations).
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    t = np.arange(n, dtype=np.float64) / sample_rate
    x = np.zeros(n, dtype=np.float64)
    for h in range(1, harmonics + 1):
        amp = 9000.0 / (h**1.1) * (0.6 + 0.4 * np.sin(2 * math.pi * 0.7 * h * t))
        if h % 5 == 0:
            amp *= 0.15
        x += amp * np.sin(2 * math.pi * fundamental * h * t + 0.3 * h)
    x += rng.normal(0.0, noise_sigma, size=n)
    if quiet_patch:
        a, b = int(1.0 * sample_rate), int(1.2 * sample_rate)
        x[a:b] *= 0.05
    peak = float(np.max(np.abs(x)))
    if peak > 0:
        x = x / peak * 0.75 * 32767.0
    return sample_rate, np.clip(np.rint(x), -32768.0, 32767.0).astype(np.int16)


def write_wav(path: Path, rate: int, samples: np.ndarray) -> Path:
    from scipy.io import wavfile

    path = Path(path)
    wavfile.write(str(path), int(rate), samples.astype(np.int16))
    return path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def cfg() -> PhaseVeilConfig:
    """Process-wide default configuration."""
    return PhaseVeilConfig()


@pytest.fixture()
def cover(tmp_path) -> Tuple[int, np.ndarray, Path]:
    """A 3 s synthetic cover written to a temp wav: ``(rate, samples, path)``."""
    rate, samples = synth_cover(seconds=3.0)
    path = write_wav(tmp_path / "cover.wav", rate, samples)
    return rate, samples, path


@pytest.fixture()
def short_cover() -> Tuple[int, np.ndarray]:
    """A 1.5 s cover (small but non-trivial capacity)."""
    return synth_cover(seconds=1.5)


@pytest.fixture()
def too_short() -> Tuple[int, np.ndarray]:
    """A 0.35 s cover that cannot hold even one RS codeword."""
    return synth_cover(seconds=0.35, quiet_patch=False)


# ---------------------------------------------------------------------------
# Network helpers
# ---------------------------------------------------------------------------


def free_udp_port() -> int:
    """Ask the OS for an unused UDP port on loopback."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return int(port)


def free_tcp_port() -> int:
    """Ask the OS for an unused TCP listening port on loopback."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    port = s.getsockname()[1]
    s.close()
    return int(port)


# ---------------------------------------------------------------------------
# Traffic generators
# ---------------------------------------------------------------------------


def natural_poisson_iats(n: int, mean_ms: float, rng: np.random.Generator) -> np.ndarray:
    """Homogeneous Poisson process: exponentially distributed IATs."""
    return rng.exponential(mean_ms, size=n)


def natural_gaussian_iats(n: int, mean_ms: float, sigma_ms: float, rng: np.random.Generator) -> np.ndarray:
    """Regular sender with Gaussian scheduling jitter (clipped positive)."""
    return np.maximum(1.0, rng.normal(mean_ms, sigma_ms, size=n))


@dataclass
class Prepared:
    """Bundle returned by the hide helpers."""

    payload: bytes
    stego: np.ndarray
    rate: int