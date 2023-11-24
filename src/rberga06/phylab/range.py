#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A custom range type."""
from dataclasses import dataclass
from math import inf as oo
from typing import Literal, final, override


def _infrepr(x: float, /) -> str:
    if x == oo:
        return "+∞"
    if x == -oo:
        return "-∞"
    return repr(x)


_PARENS = [*"[()]"]


@final
@dataclass(slots=True, frozen=True)
class Range:
    type Left  = Literal["[", "("]
    type Right = Literal["]", ")"]

    pleft:  Left  = "["
    left:   float = -oo
    right:  float = +oo
    pright: Right = "]"

    @property
    def center(self, /) -> float:
        return (self.right + self.left)/2

    # --- Constructors ---

    def __post_init__(self, /) -> None:
        if self.left == -oo:
            object.__setattr__(self, "pleft", "(")
        if self.right == +oo:
            object.__setattr__(self, "pright", ")")

    @classmethod
    def mk(cls, x: "Range | str | slice | float", /) -> "Range":
        match x:
            case Range():
                return x
            case str():
                pleft, rng, pright = x[0], x[1:-1], x[-1]
                if ";" in rng:
                    sleft, sright = map(str.strip, rng.split(";"))
                    left = float(sleft) if sleft else -oo
                    right = float(sright) if sright else +oo
                else:
                    # a single point, e.g. '[x]' => [x; x] <=> {x}
                    left = right = float(rng.strip())
                return Range(pleft, left, right, pright)  # type: ignore
            case slice():
                if x.start is None:
                    if x.step is None:
                        if x.stop is None:
                            # [:] => (-∞; +∞)
                            left  = -oo
                            right = +oo
                        else:
                            # [:y] => (-∞; y]
                            left  = -oo
                            right = x.stop
                    else:
                        # [:?:δ] => <!>
                        raise ValueError(f"Invalid slice for Range construction: {x}.")
                else:
                    if x.stop is None:
                        if x.step is None:
                            # [x:] => [x; +∞)
                            left = x.start
                            right = +oo
                        else:
                            # [x::δ] => [x-δ; x+δ]
                            left  = x.start - x.step
                            right = x.start + x.step
                    else:
                        if x.step is None:
                            # [x:y] => [x; y]
                            left  = x.start
                            right = x.stop
                        else:
                            # [x:y:δ] => <!>
                            raise ValueError(f"Invalid slice for Range construction: {x}.")
                return Range("[", left, right, "]")
            case float() | int():  # [x] => [x; x]
                return Range("[", x, x, "]")

    def __and__(self, other: "Range", /) -> "Range":
        left, pleft = max(
            (self.left, self.pleft),
            (other.left, other.pleft),
            key = lambda x: (x[0], _PARENS.index(x[1])),
        )
        right, pright = min(
            (self.right, self.pright),
            (other.right, other.pright),
            key = lambda x: (x[0], _PARENS.index(x[1])),
        )
        return Range(pleft, left, right, pright)

    def __or__(self, other: "Range", /) -> "Range":
        left, pleft = min(
            (self.left, self.pleft),
            (other.left, other.pleft),
            key = lambda x: (x[0], _PARENS.index(x[1])),
        )
        right, pright = max(
            (self.right, self.pright),
            (other.right, other.pright),
            key = lambda x: (x[0], _PARENS.index(x[1])),
        )
        return Range(pleft, left, right, pright)

    def __gt__(self, right: float, /) -> "Range":
        return self & Range("(", -oo, right, ")")

    def __ge__(self, right: float, /) -> "Range":
        return self & Range("(", -oo, right, "]")

    def __lt__(self, left: float, /) -> "Range":
        return self & Range("(", left, +oo, ")")

    def __le__(self, left: float, /) -> "Range":
        return self & Range("[", left, +oo, ")")

    def __getitem__(self, key: "Range | str | slice | float", /) -> "Range":
        return self & Range.mk(key)

    def split(self, n: int, /) -> tuple["Range", ...]:
        """Split this range into `n` smaller ones."""
        dx = (self.right - self.left)/n
        ranges = [Range("(", self.left+i*dx, self.left+(i+1)*dx, "]") for i in range(n)]
        object.__setattr__(ranges[0],  "pleft",  self.pleft)
        object.__setattr__(ranges[-1], "pright", self.pright)
        return tuple(ranges)

    # --- Python standard methods ---

    @override
    def __repr__(self, /) -> str:
        return f"{self.pleft}{_infrepr(self.left)}; {_infrepr(self.right)}{self.pright}"

    def __contains__(self, x: float, /) -> bool:
        match self.pleft:
            case "[":
                if x < self.left:
                    return False
            case "(":
                if x <= self.left:
                    return False
        match self.pright:
            case "]":
                if x > self.right:
                    return False
            case ")":
                if x >= self.right:
                    return False
        return True

    def __bool__(self, /) -> bool:
        if self.left > self.right:
            return False
        if self.left < self.right:
            return True
        if self.pleft == "(" and self.pright == ")":
            return False
        return True


R = Range()


__all__ = ["Range", "R"]
