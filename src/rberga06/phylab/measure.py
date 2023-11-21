#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Abstract measure."""
from dataclasses import dataclass
from typing import Any, Protocol, Self, final, overload


class Measure[X: (float, int)](Protocol):
    best:  X
    delta: X

    @property
    def delta_rel(self, /) -> float:
        return self.delta / self.best

    # --- Unary operators ---

    def __pos__(self, /) -> Self:
        return self

    def __neg__(self, /) -> "Measure[X]":
        return Datum(-self.best, self.delta)  # type: ignore

    # --- Binary operators ---

    @overload
    def __add__[T: (float, int)](self: "Measure[int]", other: "Measure[T] | T", /) -> "Measure[T]": ...
    @overload
    def __add__(self: "Measure[float]", other: "Measure[float] | Measure[int] | float | int", /) -> "Measure[float]": ...
    def __add__(self, other: "Measure[Any] | float", /) -> "Measure[Any]":
        if isinstance(other, float | int):
            return Datum(self.best + other, self.delta)
        return Datum(self.best + other.best, self.delta + other.delta)

    @overload
    def __sub__[T: (float, int)](self: "Measure[int]", other: "Measure[T] | T", /) -> "Measure[T]": ...
    @overload
    def __sub__(self: "Measure[float]", other: "Measure[float] | Measure[int] | float | int", /) -> "Measure[float]": ...
    def __sub__(self, other: "Measure[Any] | float", /) -> "Measure[Any]":
        return self + (-other)  # type: ignore

    @overload
    def __mul__[T: (float, int)](self: "Measure[int]", other: "Measure[T] | T", /) -> "Measure[T]": ...
    @overload
    def __mul__(self: "Measure[float]", other: "Measure[float] | Measure[int] | float | int", /) -> "Measure[float]": ...
    def __mul__(self, other: "Measure[Any] | float", /) -> "Measure[Any]":
        if isinstance(other, float | int):
            return Datum(self.best * other, self.delta * abs(other))
        return Datum.from_delta_rel(self.best * other.best, self.delta_rel + other.delta_rel)

    def __truediv__(self, other: "Measure[float] | Measure[int] | float", /) -> "Measure[float]":
        if isinstance(other, float | int):
            return Datum(self.best / other, self.delta / abs(other))
        return Datum.from_delta_rel(self.best / other.best, self.delta_rel + other.delta_rel)

    def __pow__(self, other: int, /) -> "Measure[X]":
        return Datum.from_delta_rel(self.best ** other, self.delta_rel * abs(other))

    # --- Right operands ---

    @overload
    def __radd__[T: (float, int)](self: "Measure[int]", other: "Measure[T] | T", /) -> "Measure[T]": ...
    @overload
    def __radd__(self: "Measure[float]", other: "Measure[float] | Measure[int] | float | int", /) -> "Measure[float]": ...
    def __radd__(self, other: "Measure[Any] | float", /) -> "Measure[Any]":
        return self + other  # type: ignore

    @overload
    def __rsub__[T: (float, int)](self: "Measure[int]", other: "Measure[T] | T", /) -> "Measure[T]": ...
    @overload
    def __rsub__(self: "Measure[float]", other: "Measure[float] | Measure[int] | float | int", /) -> "Measure[float]": ...
    def __rsub__(self, other: "Measure[Any] | float", /) -> "Measure[Any]":
        return (-self) + other  # type: ignore

    @overload
    def __rmul__[T: (float, int)](self: "Measure[int]", other: "Measure[T] | T", /) -> "Measure[T]": ...
    @overload
    def __rmul__(self: "Measure[float]", other: "Measure[float] | Measure[int] | float | int", /) -> "Measure[float]": ...
    def __rmul__(self, other: "Measure[Any] | float", /) -> "Measure[Any]":
        return self * other  # type: ignore

    def __rtruediv__(self, other: "Measure[float] | Measure[int] | float", /) -> "Measure[float]":
        if isinstance(other, float | int):
            return Datum.from_delta_rel(other / self.best, self.delta_rel)
        return Datum.from_delta_rel(other.best / self.best, other.delta_rel + self.delta_rel)

    # --- Comparison ---

    def ε(self, other: "Measure[float] | Measure[int] | float", /) -> float:
        if isinstance(other, float | int):
            return (self.best - other)/self.delta
        return (self.best - other.best)/(self.delta + other.delta)


@final
@dataclass(slots=True, frozen=True)
class Datum[X: (float, int)](Measure[X]):
    best:  X
    delta: X

    @classmethod
    def from_const(cls, const: float, /) -> Self:
        return cls(const, 0)

    @classmethod
    def from_delta_rel(cls, best: float, delta_rel: float, /) -> Self:
        return cls(best, delta_rel * best)


type AMeasure = Measure[float] | Measure[int]

__all__ = ["Measure", "Datum"]
