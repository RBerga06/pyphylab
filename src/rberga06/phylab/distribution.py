#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from dataclasses import dataclass
from math import ceil, floor
from typing import Any, Iterable, Iterator, Protocol, Self, Sequence, final, overload, override
from typing_extensions import deprecated
from .measure import Measure, MeasureLike, best
from .range import Range
from .bins import ADataSet, BinSet


@final
@dataclass(frozen=True, slots=True)
class DistFit[D: "Distribution[Any]", S: ADataSet[MeasureLike[float]], X: MeasureLike[float]]:
    dist: D
    data: S
    bins: BinSet[S, X]


class Distribution[T: float](Protocol):
    """An (abstract) distribution."""

    def pdf(self, x: T, /) -> float:
        """The probability density function (PDF), evaluated at x."""
        ...

    def p(self, x1: T, x2: T, /) -> float:
        """The probability of getting something between x1 and x2."""
        ...

    def p_worse(self, x: T, /) -> float:
        """The probability of getting x, or worse."""
        ...

    def chauvenet(self, n: int, x: T, /) -> bool:
        """Check if we can apply Chauvenet to x."""
        return (n * self.p_worse(x)) < .5

    def expected(self, n: int, x1: T, x2: T, /) -> float:
        """The expected value between `x1` and `x2`."""
        return self.p(x1, x2) * n

    def sample(self, n: int, /) -> tuple[float, ...]:
        """Return pseudo-random data with this distribution."""
        raise NotImplementedError

    @classmethod
    def fit[S: ADataSet[MeasureLike[float]]](cls, data: S, /) -> DistFit[Self, S, MeasureLike[float]]:
        """Find the distribution that best fits `data`."""
        ...


# --- OLD CODE BELOW ---


def _cumulative[X](data: Iterable[X], /) -> Iterator[tuple[X, ...]]:
    cumul: tuple[X, ...] = ()
    yield cumul
    for x in data:
        cumul += (x,)
        yield cumul


class _DataSequence[M: MeasureLike[float]](Protocol):
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
@deprecated("Use `Bin` instead.")
class DistBin[M: MeasureLike[float]](_DataSequence[M]):
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



@deprecated("Use `Distribution` instead.")
class Dist[M: MeasureLike[float]](_DataSequence[M], Measure[float], Protocol):
    """An (abstract) distribution."""
    data: Sequence[M]

    # --- Bins ---
    custom_bins_start: float | None = None
    custom_bins_stop:  float | None = None

    @property
    def bins_start(self, /) -> float:
        if self.custom_bins_start is None:
            return min(map(best, self.data))
        return self.custom_bins_start

    @property
    def bins_stop(self, /) -> float:
        if self.custom_bins_stop is None:
            return max(map(best, self.data))
        return self.custom_bins_stop

    @property
    def bins_ranges(self, /) -> Iterable[Range]: ...

    @property
    def bins(self, /) -> tuple[DistBin[M], ...]:
        return tuple([DistBin(self, r) for r in self.bins_ranges])

    # --- Statistics on data ---

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
            return tuple([self.probability(r) * n for r in self.bins_ranges])
        return self.probability(x) * len(self.data)

    @property
    def expected_avg(self, /) -> float:
        expected = self.expected()
        return sum(expected)/len(expected)

    @classmethod
    def mk_iter_cumulative(
        cls,
        data: Iterable[M],
        /, *,
        custom_bins_start: float | None = None,
        custom_bins_stop:  float | None = None,
    ) -> Iterator[Self]:
        for d in _cumulative(data):
            yield cls(d, custom_bins_start=custom_bins_start, custom_bins_stop=custom_bins_stop)



@deprecated("Use `Distribution` instead.")
class DiscreteDist[M: Measure[int] | int](Dist[M], Protocol):
    """A discrete (abstract) distribution."""

    def discrete_probability(self, x: int, /) -> float: ...

    @override
    def probability(self, x: Range, /) -> float:
        return sum(map(self.discrete_probability, range(int(ceil(x.left)), int(floor(x.right)) + 1)))


__all__ = ["DistBin", "Dist", "DiscreteDist"]
