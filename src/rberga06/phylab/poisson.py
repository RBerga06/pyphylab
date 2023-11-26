#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Poisson distribution."""
from math import factorial, exp, sqrt
from dataclasses import dataclass
from typing import Sequence, override

from .range import Range
from .measure import MeasureLike
from .distribution import DiscreteDist


@dataclass(slots=True, frozen=True)
class Poisson[M: MeasureLike[int]](DiscreteDist[M]):
    data: Sequence[M]

    @property
    @override
    def binsr(self, /) -> tuple[Range, ...]:
        if not self.data and None in (self.custom_bins_start, self.custom_bins_stop):
            return ()
        return tuple(map(Range.mk, range(int(self.bins_start), int(self.bins_stop) + 1)))

    @property
    def sigma(self, /) -> float:
        return sqrt(self.average)

    @property
    @override
    def best(self, /) -> float:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.average

    @property
    @override
    def delta(self, /) -> float:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.sigma / sqrt(len(self.data))

    @override
    def discrete_probability(self, x: int, /) -> float:
        µ = self.average
        return (pow(µ, x) * exp(-µ))/factorial(x)


__all__ = ["Poisson"]
