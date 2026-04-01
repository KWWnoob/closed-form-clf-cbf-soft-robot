"""CBFpy: Control Barrier Functions in Python and Jax"""

import jax as _jax

# 64 bit precision is generally necessary for these problems to be feasible
_jax.config.update("jax_enable_x64", True)

from cbfpy.cbfs.cbf import CBF
from cbfpy.cbfs.clf_cbf import CLFCBF
from cbfpy.config.cbf_config import CBFConfig
from cbfpy.config.clf_cbf_config import CLFCBFConfig
