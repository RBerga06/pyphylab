#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Normal distribution."""
from math import erf, floor, sqrt
from dataclasses import dataclass
from typing import Sequence, override

from .range import Range
from .measure import Measure
from .distribution import Dist


@dataclass(slots=True, frozen=True)
class Normal[M: Measure](Dist[M]):
    data: Sequence[M]

    @property
    def average(self, /) -> float:
        return sum([len(bin) * bin.center for bin in self.bins])

    @property
    def sigma(self, /) -> float:
        µ = self.average
        N = len(self.data)
        return sqrt(sum([len(b)*(b.center - µ)**2 for b in self.bins])/(N-1))

    @property
    def nbins(self, /) -> int:
        return int(floor(sqrt(len(self.data))))

    @property
    @override
    def best(self, /) -> float:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.average

    @property
    @override
    def delta(self, /) -> float:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.sigma / sqrt(len(self.data))

    @override
    def probability(self, r: Range, /) -> float:
        µ, s = self.average, self.sigma
        return (erf((r.right-µ)/s) - erf((r.left-µ)/s))/2

    @property
    @override
    def binsr(self, /) -> tuple[Range, ...]:
        left  = min(self.data, key=lambda m: m.best)
        right = max(self.data, key=lambda m: m.best)
        return Range("[", left.best - left.delta, right.best + right.delta, "]").split(self.nbins)
