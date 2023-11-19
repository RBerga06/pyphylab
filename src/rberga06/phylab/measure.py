#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Abstract measure."""
from dataclasses import dataclass
from typing import Protocol, Self, final


class Measure(Protocol):
    best: float
    delta: float

    @property
    def delta_rel(self, /) -> float:
        return self.delta / self.best

    # --- Unary operators ---

    def __pos__(self, /) -> Self:
        return self

    def __neg__(self, /) -> "Measure":
        return Datum(-self.best, self.delta)

    # --- Binary operators ---

    def __add__(self, other: "Measure | float") -> "Measure":
        if isinstance(other, float | int):
            return Datum(self.best + other, self.delta)
        return Datum(self.best + other.best, self.delta + other.delta)

    def __sub__(self, other: "Measure | float", /) -> "Measure":
        return self + (-other)

    def __mul__(self, other: "Measure | float", /) -> "Measure":
        if isinstance(other, float | int):
            return Datum(self.best * other, self.delta * abs(other))
        return Datum.from_delta_rel(self.best * other.best, self.delta_rel + other.delta_rel)

    def __truediv__(self, other: "Measure | float", /) -> "Measure":
        if isinstance(other, float | int):
            return Datum(self.best / other, self.delta / abs(other))
        return Datum.from_delta_rel(self.best / other.best, self.delta_rel + other.delta_rel)

    def __pow__(self, other: int, /) -> "Measure":
        return Datum.from_delta_rel(self.best ** other, self.delta_rel * abs(other))

    # --- Right operands ---

    def __radd__(self, other: "Measure | float", /) -> "Measure":
        return self + other

    def __rsub__(self, other: "Measure | float", /) -> "Measure":
        return (-self) + other

    def __rmul__(self, other: "Measure | float", /) -> "Measure":
        return self * other

    def __rtruediv__(self, other: "Measure | float", /) -> "Measure":
        if isinstance(other, float | int):
            return Datum.from_delta_rel(other / self.best, self.delta_rel)
        return Datum.from_delta_rel(other.best / self.best, other.delta_rel + self.delta_rel)

    # --- Comparison ---

    def ε(self, other: "Measure | float", /) -> float:
        if isinstance(other, float | int):
            return (self.best - other)/self.delta
        return (self.best - other.best)/(self.delta + other.delta)


@final
@dataclass(slots=True, frozen=True)
class Datum(Measure):
    best: float
    delta: float

    @classmethod
    def from_const(cls, const: float, /) -> Self:
        return cls(const, 0)

    @classmethod
    def from_delta_rel(cls, best: float, delta_rel: float, /) -> Self:
        return cls(best, delta_rel * best)


__all__ = ["Measure", "Datum"]
