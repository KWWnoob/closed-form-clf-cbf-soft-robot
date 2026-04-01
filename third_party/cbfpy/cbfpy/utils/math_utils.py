"""Assorted helper functions related to math operations / linear algebra"""

import jax.numpy as jnp
from jax import Array
from jax.typing import ArrayLike


def normalize(vec: ArrayLike) -> Array:
    """Normalizes a vector to have magnitude 1

    If normalizing an array of vectors, each vector will have magnitude 1

    Args:
        vec (ArrayLike): Input vector or array. Shape (dim,) or (n_vectors, dim)

    Returns:
        Array: Unit vector(s), shape (dim,) or (n_vectors, dim) (same shape as the input)
    """
    vec = jnp.atleast_1d(vec)
    norms = jnp.linalg.norm(vec, axis=-1)
    return vec / norms[..., jnp.newaxis]
