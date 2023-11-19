#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol, Sequence, final, overload, override
from .measure import Measure
from .range import Range


class _MeasureSequence[M: Measure](Protocol):
    """A proxy around the `data` attribute."""
    data: Sequence[M]

    def __len__(self, /) -> int:
        return len(self.data)

    def __iter__(self, /) -> Iterator[M]:
        return iter(self.data)

    def __getitem__(self, key: "Range | str | slice | float", /) -> tuple[M, ...]:
        return tuple([x for x in self.data if x.best in Range.mk(key)])



@final
@dataclass(slots=True, frozen=True)
class DistBin[M: Measure](_MeasureSequence[M]):
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



class Dist[M: Measure](_MeasureSequence[M], Measure, Protocol):
    """An (abstract) distribution."""
    data: Sequence[M]

    @property
    def binsr(self, /) -> Iterable[Range]: ...

    @property
    def bins(self, /) -> tuple[DistBin[M], ...]:
        return tuple([DistBin(self, r) for r in self.binsr])

    def probability(self, x: Range, /) -> float: ...

    @overload
    def expected(self, x: None = ..., /) -> float: ...
    @overload
    def expected(self, x: Range, /) -> float: ...
    def expected(self, x: Range | None = None, /) -> float | tuple[float, ...]:
        if x is None:
            n = len(self.data)
            return tuple([self.probability(r)*n for r in self.binsr])
        return self.probability(x) * len(self.data)



__all__ = ["DistBin", "Dist"]
