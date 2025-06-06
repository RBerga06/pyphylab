#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ruff: noqa: F403
"""\
`rberga06.phylab`: a Python package for daily needs at the Physics lab.
Developed with ❤️ by @RBerga06, Physics student @ UniPR (Parma, Italy).\
"""

# TODO: Once we have a fully-working Rust backend switch this with `from ._lib import *`
from ._tensor_numpy import *

# Core data structures & utilities
from .measure import *
from .data import *
from .bins import *
from .distribution import *
from .dataset import *

# Distributions
from .normal import *
from .poisson import *
from .bernoulli import *

# Constants
from .constants import *
