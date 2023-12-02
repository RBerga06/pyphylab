#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Physics constants."""
# Imports are '_'-prefixed to avoid auto exporting
#   (__all__ is not defined, since we might define several constants).
import math as _m
import typing as _t
from .measure import Datum as _Datum, MeasureLike as _MeasureLike

def _y2s[T: _MeasureLike[float]](t: T, /) -> T:
    """Convert t from years to SI seconds."""
    # 1y = 365d 6h 9'10" | source: Wikipedia
    return t * 365.256363051 * 24 * 60 * 60  # type: ignore

class _Isotope(_t.NamedTuple):
    N: int
    """Atomic mass number"""
    Z: int
    """Atomic number"""
    T12: _MeasureLike[float]
    """Half-life"""
    @property
    def mass(self, /) -> float:
        """The mass of one atom"""
        return self.Z * (M_proton + M_electron) + (self.N - self.Z) * M_neutron

# Mathematical constants
ln2 = _m.log(2)
π = _m.pi
# Physical constants
g = _Datum(9.806, 0.001)  # kg m s⁻²
avogadro = 6.02214076e23  # mol⁻¹
M_proton  = 1.67262192369e-27 # kg
M_neutron = 1.67492749804e-27 # kg
M_electron = 9.1093837015e-31 # kg
# Isotopes
Th232 = _Isotope(232, 90, _y2s(_Datum(1.405, .001)*1e10))
