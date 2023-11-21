#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from math import ceil, floor
from typing import Iterable, Iterator, Protocol, Self, Sequence, final, overload, override
from .measure import Measure, AMeasure
from .range import Range


def _cumulative[X](data: Iterable[X]) -> Iterator[tuple[X, ...]]:
    cumul: tuple[X, ...] = ()
    yield cumul
    for x in data:
        cumul += (x,)
        yield cumul


def best[X: (float, int)](x: Measure[X] | X, /) -> X:
    if isinstance(x, float | int):
        return x
    return x.best


class _DataSequence[M: AMeasure | float](Protocol):
    """A proxy around the `data` attribute."""
    data: Sequence[M]

    def __len__(self, /) -> int:
        return len(self.data)

    def __iter__(self, /) -> Iterator[M]:
        return iter(self.data)

    def __getitem__(self, key: "Range | str | slice | float", /) -> tuple[M, ...]:
        return tuple([x for x in self.data if best(x) in Range.mk(key)])


@final
@dataclass(slots=True, frozen=True)
class DistBin[M: AMeasure | float](_DataSequence[M]):
    """A dynamic container for distribution bins."""
    dist: "Dist[M]"
    r: Range

    @property
    def center(self, /) -> float:
        return (self.r.right + self.r.left)/2

    @property
    @override
    def data(self, /) -> tuple[M, ...]:  # pyright: ignore[reportIncompatibleVariableOverride]
        return self.dist[self.r]



class Dist[M: AMeasure | float](_DataSequence[M], Measure[float], Protocol):
    """An (abstract) distribution."""
    data: Sequence[M]

    @property
    def binsr(self, /) -> Iterable[Range]: ...

    @property
    def bins(self, /) -> tuple[DistBin[M], ...]:
        return tuple([DistBin(self, r) for r in self.binsr])

    @property
    def average(self, /) -> float:
        if not self.data:
            return 0.
        return sum([len(bin) * bin.center for bin in self.bins])/len(self.data)

    def probability(self, x: Range, /) -> float: ...

    @overload
    def expected(self, x: None = ..., /) -> tuple[float, ...]: ...
    @overload
    def expected(self, x: Range, /) -> float: ...
    def expected(self, x: Range | None = None, /) -> float | tuple[float, ...]:
        if x is None:
            n = len(self.data)
            return tuple([self.probability(r)*n for r in self.binsr])
        return self.probability(x) * len(self.data)

    @classmethod
    def mk_iter_cumulative(cls, data: Iterable[M]) -> Iterator[Self]:
        for d in _cumulative(data):
            yield cls(d)



class DiscreteDist[M: Measure[int] | int](Dist[M], Protocol):
    """A discrete (abstract) distribution."""

    def discrete_probability(self, x: int, /) -> float: ...

    @override
    def probability(self, x: Range, /) -> float:
        return sum(map(self.discrete_probability, range(int(ceil(x.left)), int(floor(x.right)))))



__all__ = ["DistBin", "Dist", "DiscreteDist"]
