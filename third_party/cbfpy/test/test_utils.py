"""Unit tests for utility functions"""

import unittest
import numpy as np
import jax

import cbfpy.utils.math_utils as math_utils


TEST_JIT = False


class TestUtils(unittest.TestCase):
    """Unit tests for utility functions"""

    def test_normalize(self):
        if TEST_JIT:
            normalize = jax.jit(math_utils.normalize)
        else:
            normalize = math_utils.normalize

        # Test single vector
        vec = np.array([1, 2, 3])
        self.assertTrue(np.allclose(normalize(vec), vec / np.linalg.norm(vec)))

        # Test multiple vectors
        vecs = np.array([[1, 2, 3], [4, 5, 6]])
        normalized = normalize(vecs)
        for i, vec in enumerate(vecs):
            self.assertTrue(np.allclose(normalized[i], vec / np.linalg.norm(vec)))


if __name__ == "__main__":
    unittest.main()
