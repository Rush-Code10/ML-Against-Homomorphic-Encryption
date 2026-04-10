from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import numpy as np


try:
    import tenseal as ts
except ImportError:  # pragma: no cover
    ts = None


BackendName = Literal["tenseal", "mock"]


@dataclass(slots=True)
class FHEContext:
    backend: BackendName
    context: Any | None
    poly_modulus_degree: int = 8192
    coeff_mod_bit_sizes: list[int] = field(default_factory=lambda: [60, 30, 30, 30, 60])
    global_scale: float = 2**30
    slots: int = 16


@dataclass(slots=True)
class EncryptedTensor:
    data: Any
    backend: BackendName
    size_hint: int
    depth: int = 0
    operation_count: int = 0
    noise_hint: float = 100.0
    length: int = 0


def init_context(
    vector_size: int = 16,
    backend_preference: BackendName = "tenseal",
) -> FHEContext:
    if backend_preference == "tenseal" and ts is not None:
        context = ts.context(
            ts.SCHEME_TYPE.CKKS,
            poly_modulus_degree=8192,
            coeff_mod_bit_sizes=[60, 30, 30, 30, 60],
        )
        context.global_scale = 2**30
        context.auto_relin = True
        context.auto_rescale = True
        context.auto_mod_switch = True
        context.generate_galois_keys()
        context.generate_relin_keys()
        return FHEContext(
            backend="tenseal",
            context=context,
            coeff_mod_bit_sizes=[60, 30, 30, 30, 60],
            global_scale=2**30,
            slots=vector_size,
        )

    return FHEContext(
        backend="mock",
        context=None,
        slots=vector_size,
    )


def _as_numpy(vector: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(vector), dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Only one-dimensional vectors are supported.")
    return array


def encrypt_vector(context: FHEContext, vector: Iterable[float]) -> EncryptedTensor:
    array = _as_numpy(vector)
    if context.backend == "tenseal":
        encrypted = ts.ckks_vector(context.context, array.tolist())
        size_hint = max(len(encrypted.serialize()), array.size * 8)
    else:
        encrypted = array
        size_hint = array.size * 8 + 128

    return EncryptedTensor(
        data=encrypted,
        backend=context.backend,
        size_hint=size_hint,
        length=int(array.size),
        noise_hint=100.0,
    )


def decrypt_vector(context: FHEContext, encrypted: EncryptedTensor) -> np.ndarray:
    if encrypted.backend != context.backend:
        raise ValueError("Encrypted tensor backend does not match context backend.")
    if context.backend == "tenseal":
        return np.asarray(encrypted.data.decrypt(), dtype=np.float64)
    return np.asarray(encrypted.data, dtype=np.float64)
