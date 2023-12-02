#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physics constants."""
# Imports are '_'-prefixed to avoid auto exporting
#   (__all__ is not defined, since we might define several constants).
import math as _m
from .measure import Datum as _Datum

# Mathematical constants
ln2 = _m.log(2)
π = _m.pi
# Physical constants
g = _Datum(9.806, 0.001)
avogadro = 6.02214076e23
