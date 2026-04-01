"""Jax helper functions and decorators"""

import os

import jax


def conditional_jit(condition: bool):
    """Decorator to jit a function if a condition is met.

    Args:
        condition (bool): True to jit the function, False otherwise
    """

    def wrapper(func):
        if condition:
            return jax.jit(func)
        return func

    return wrapper


def jit_if_not_debugging(func):
    """Decorator to jit a function if the DEBUG environment variable is not set."""
    debug = os.environ.get("DEBUG", "").lower() in ("1", "true")
    return conditional_jit(not debug)(func)
