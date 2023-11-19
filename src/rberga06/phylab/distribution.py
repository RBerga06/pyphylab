#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from typing import Iterable, Iterator, Protocol, Sequence, final, overload, override
from .measure import Measure
from .range import Range


class _ImmutableDataProxy[T](Protocol):
    """A proxy around the `data` attribute."""
    data: Sequence[T]

    def __len__(self, /) -> int:
        return len(self.data)

    def __iter__(self, /) -> Iterator[T]:
        return iter(self.data)

    @overload
    def __getitem__(self, key: int, /) -> T: ...
    @overload
    def __getitem__(self, key: slice, /) -> Sequence[T]: ...
    def __getitem__(self, key: int | slice, /) -> T | Sequence[T]:
        return self.data[key]



@final
@dataclass(slots=True, frozen=True)
class DistBin[M: Measure](_ImmutableDataProxy[M]):
    """A dynamic container for distribution bins."""
    dist: "Dist[M]"
    r: Range

    @property
    def center(self, /) -> float:
        return (self.r.right + self.r.left)/2

    @property
    @override
    def data(self, /) -> tuple[M, ...]:  # pyright: ignore[reportIncompatibleVariableOverride]
        return tuple([d for d in self.dist.data if d.best in self.r])



class Dist[M: Measure](_ImmutableDataProxy[M], Measure, Protocol):
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
