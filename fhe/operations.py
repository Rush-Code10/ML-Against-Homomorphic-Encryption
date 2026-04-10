from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np

from fhe.fhe_setup import EncryptedTensor, FHEContext, encrypt_vector


def _match_shapes(left: EncryptedTensor, right: EncryptedTensor) -> None:
    if left.length != right.length:
        raise ValueError("Encrypted tensors must have the same logical length.")


def _binary_size(left: EncryptedTensor, right: EncryptedTensor) -> int:
    return max(left.size_hint, right.size_hint)


def _combine_noise(left: EncryptedTensor, right: EncryptedTensor, penalty: float) -> float:
    return max(1.0, min(left.noise_hint, right.noise_hint) - penalty)


def encrypted_add(left: EncryptedTensor, right: EncryptedTensor) -> EncryptedTensor:
    _match_shapes(left, right)
    data = left.data + right.data
    return EncryptedTensor(
        data=data,
        backend=left.backend,
        size_hint=_binary_size(left, right),
        depth=max(left.depth, right.depth),
        operation_count=left.operation_count + right.operation_count + 1,
        noise_hint=_combine_noise(left, right, 0.2),
        length=left.length,
    )


def encrypted_sub(left: EncryptedTensor, right: EncryptedTensor) -> EncryptedTensor:
    _match_shapes(left, right)
    data = left.data - right.data
    return EncryptedTensor(
        data=data,
        backend=left.backend,
        size_hint=_binary_size(left, right),
        depth=max(left.depth, right.depth),
        operation_count=left.operation_count + right.operation_count + 1,
        noise_hint=_combine_noise(left, right, 0.25),
        length=left.length,
    )


def encrypted_mul(left: EncryptedTensor, right: EncryptedTensor) -> EncryptedTensor:
    _match_shapes(left, right)
    data = left.data * right.data
    return EncryptedTensor(
        data=data,
        backend=left.backend,
        size_hint=_binary_size(left, right) + 32,
        depth=max(left.depth, right.depth) + 1,
        operation_count=left.operation_count + right.operation_count + 1,
        noise_hint=_combine_noise(left, right, 4.0),
        length=left.length,
    )


def encrypted_scalar_mul(ciphertext: EncryptedTensor, scalar: float) -> EncryptedTensor:
    data = ciphertext.data * scalar
    return replace(
        ciphertext,
        data=data,
        operation_count=ciphertext.operation_count + 1,
        noise_hint=max(1.0, ciphertext.noise_hint - 1.5),
    )


def encrypted_sum(ciphertext: EncryptedTensor) -> EncryptedTensor:
    if ciphertext.backend == "tenseal":
        try:
            summed = ciphertext.data.sum()
            if hasattr(summed, "serialize"):
                data = summed
            else:
                data = ciphertext.data * 0 + float(summed)
        except AttributeError:
            data = ciphertext.data
    else:
        data = np.asarray([np.sum(ciphertext.data)], dtype=np.float64)

    return EncryptedTensor(
        data=data,
        backend=ciphertext.backend,
        size_hint=max(64, ciphertext.size_hint // max(1, ciphertext.length)),
        depth=ciphertext.depth,
        operation_count=ciphertext.operation_count + max(ciphertext.length - 1, 1),
        noise_hint=max(1.0, ciphertext.noise_hint - 0.75),
        length=1,
    )


def encrypted_mean(ciphertext: EncryptedTensor) -> EncryptedTensor:
    summed = encrypted_sum(ciphertext)
    return encrypted_scalar_mul(summed, 1.0 / max(ciphertext.length, 1))


def encrypted_variance(ciphertext: EncryptedTensor) -> EncryptedTensor:
    mean_value = encrypted_mean(ciphertext)
    squared_values = encrypted_mul(ciphertext, ciphertext)
    mean_of_squares = encrypted_mean(squared_values)
    squared_mean = encrypted_mul(mean_value, mean_value)
    return encrypted_sub(mean_of_squares, squared_mean)


def encrypted_dot_product(left: EncryptedTensor, right: EncryptedTensor) -> EncryptedTensor:
    return encrypted_sum(encrypted_mul(left, right))


def _encrypt_scalar_like(reference: EncryptedTensor, value: float) -> EncryptedTensor:
    if reference.backend == "tenseal":
        data = reference.data + value
    else:
        data = np.asarray([value], dtype=np.float64)
    return EncryptedTensor(
        data=data,
        backend=reference.backend,
        size_hint=reference.size_hint,
        length=1,
    )


def encrypted_linear_regression_inference(
    features: EncryptedTensor,
    weights: EncryptedTensor,
    bias: float,
) -> EncryptedTensor:
    score = encrypted_dot_product(features, weights)
    if bias == 0.0:
        return encrypted_scalar_mul(score, 1.0)
    if score.backend == "tenseal":
        return EncryptedTensor(
            data=score.data + bias,
            backend=score.backend,
            size_hint=score.size_hint,
            depth=score.depth,
            operation_count=score.operation_count + 1,
            noise_hint=max(1.0, score.noise_hint - 0.2),
            length=score.length,
        )
    return encrypted_add(score, _encrypt_scalar_like(score, bias))


def encrypted_polynomial_sigmoid(ciphertext: EncryptedTensor) -> EncryptedTensor:
    if ciphertext.backend == "tenseal":
        x = ciphertext
        x2 = encrypted_mul(x, x)
        linear = encrypted_scalar_mul(x, 0.197)
        quadratic = encrypted_scalar_mul(x2, -0.004)
        partial = encrypted_add(linear, quadratic)
        return EncryptedTensor(
            data=partial.data + 0.5,
            backend=partial.backend,
            size_hint=partial.size_hint,
            depth=partial.depth,
            operation_count=partial.operation_count + 1,
            noise_hint=max(1.0, partial.noise_hint - 0.2),
            length=partial.length,
        )

    x = ciphertext
    x2 = encrypted_mul(x, x)
    x3 = encrypted_mul(x2, x)
    term1 = encrypted_scalar_mul(x, 0.25)
    term2 = encrypted_scalar_mul(x3, -1.0 / 48.0)
    partial = encrypted_add(term1, term2)
    if partial.backend == "tenseal":
        return EncryptedTensor(
            data=partial.data + 0.5,
            backend=partial.backend,
            size_hint=partial.size_hint,
            depth=partial.depth,
            operation_count=partial.operation_count + 1,
            noise_hint=max(1.0, partial.noise_hint - 0.2),
            length=partial.length,
        )
    constant = _encrypt_scalar_like(ciphertext, 0.5)
    return encrypted_add(constant, partial)


def encrypted_logistic_regression_approx(
    features: EncryptedTensor,
    weights: EncryptedTensor,
    bias: float,
) -> EncryptedTensor:
    return encrypted_polynomial_sigmoid(
        encrypted_linear_regression_inference(features, weights, bias)
    )


def encrypt_numpy(context: FHEContext, array: Sequence[float]) -> EncryptedTensor:
    return encrypt_vector(context, np.asarray(array, dtype=np.float64))
