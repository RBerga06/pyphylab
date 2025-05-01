# Implementation of the `_tensor.pyi` API baked buy numpy.
# pyright: reportAny = false
from dataclasses import dataclass
from typing_extensions import TypeVar, TypeVarTuple, Generic, Self, final, overload
from collections.abc import Iterable, Iterator
import numpy as np

type Elem = (
    # Simd[Elem, *tuple[Any, ...]] |  # TODO
    # Mat[int, int, Elem] | Vec[int, Elem] |  # TODO
    bool | int | float
)
_N_co = TypeVar("_N_co", bound=int, covariant=True)
_M_co = TypeVar("_M_co", bound=int, covariant=True)
_T_co = TypeVar("_T_co", bound=Elem, covariant=True)
_S = TypeVarTuple("_S")


@final
@dataclass(frozen=True, slots=True)
class Vec(Generic[_N_co, _T_co]):
    """A 1D vector."""

    _n: _N_co
    _np: np.ndarray[tuple[_N_co], np.dtype[np.object_]]

    def __init__(self, it: Iterable[_T_co], n: _N_co | None = None, /) -> None:
        object.__setattr__(self, "_np", np.array(it))
        object.__setattr__(self, "_len", self._np.shape[0])
        if n is not None:
            assert self._n == n

    @property
    def shape(self, /) -> tuple[_N_co]:
        return (self._n,)

    def __add__(self, rhs: Self | _T_co, /) -> Self:
        if isinstance(rhs, Vec):
            return type(self)(self._np + rhs._np, self._n)
        return type(self)(self._np + rhs, self._n)

    def __sub__(self, rhs: Self | _T_co, /) -> Self:
        if isinstance(rhs, Vec):
            return type(self)(self._np - rhs._np, self._n)
        return type(self)(self._np - rhs, self._n)

    def __rsub__(self, lhs: Self | _T_co, /) -> Self:
        if isinstance(lhs, Vec):
            return type(self)(lhs._np - self._np, self._n)
        return type(self)(lhs - self._np, self._n)

    def __mul__(self, rhs: Self | _T_co, /) -> Self:
        if isinstance(rhs, Vec):
            return type(self)(self._np * rhs._np, self._n)
        return type(self)(self._np * rhs, self._n)

    def __truediv__(self, rhs: Self | _T_co, /) -> Self:
        if isinstance(rhs, Vec):
            return type(self)(self._np / rhs._np, self._n)
        return type(self)(self._np / rhs, self._n)

    def __rtruediv__(self, lhs: Self | _T_co, /) -> Self:
        if isinstance(lhs, Vec):
            return type(self)(lhs._np / self._np, self._n)
        return type(self)(lhs / self._np, self._n)

    def __pow__(self, rhs: Self | _T_co, /) -> Self:
        if isinstance(rhs, Vec):
            return type(self)(self._np**rhs._np, self._n)
        return type(self)(self._np**rhs, self._n)

    def __rpow__(self, lhs: Self | _T_co, /) -> Self:
        if isinstance(lhs, Vec):
            return type(self)(lhs._np**self._np, self._n)
        return type(self)(lhs**self._np, self._n)

    __radd__ = __add__
    __rmul__ = __mul__

    def __matmul__(self, rhs: Self, /) -> _T_co:
        """This is just the standard dot product of two vectors."""
        return self._np.dot(rhs._np)[()]

    __rmatmul__ = __matmul__

    def __iter__(self, /) -> Iterator[_T_co]: ...


@final
class Mat(Generic[_N_co, _M_co, _T_co]):
    """A 2D matrix (N rows, M columns)."""

    @property
    def shape(self, /) -> tuple[_N_co, _M_co]: ...
    def __iter__(self, /) -> Iterator[Vec[_M_co, _T_co]]: ...
    def __add__(self, rhs: Self | _T_co, /) -> Self: ...

    __radd__ = __pow__ = __truediv__ = __mul__ = __sub__ = __add__
    __rpow__ = __rsub__ = __rmul__ = __rtruediv__ = __radd__

    @overload
    def __matmul__[K: int](
        self, rhs: "Mat[_M_co, K, _T_co]", /
    ) -> "Mat[_N_co, K, _T_co]": ...
    @overload
    def __matmul__(self, rhs: Vec[_M_co, _T_co], /) -> Vec[_N_co, _T_co]: ...
    def __rmatmul__(self, lhs: Vec[_N_co, _T_co], /) -> Vec[_M_co, _T_co]: ...
    @property
    def T(self, /) -> "Mat[_M_co, _N_co, _T_co]":
        """Transpose this matrix."""
        ...

    @property
    def into_vec[X: Elem](
        self: "Mat[_N_co, _M_co, X]", /
    ) -> "Vec[_N_co, Vec[_M_co, X]]":
        """Reinterpret `self` as a vector of vectors (rows)"""
        ...

    # @property
    # def into_simd(self, /) -> "Simd[_T_co, _N_co, _M_co]":
    #     """Reinterpret `self` as a simd"""
    #     ...

    def rk(self, /) -> int:
        """Evaluate the rank of this matrix."""
        ...

    def det[N: int](self: "Mat[N, N, _T_co]", /) -> _T_co:
        """Evaluate the determinant of this matrix."""
        ...

    def inv[N: int](self: "Mat[N, N, _T_co]", /) -> "Mat[N, N, _T_co]":
        """Evaluate the inverse of this matrix."""
        ...

    def mat_exp[N: int](self: "Mat[N, N, _T_co]", /) -> "Mat[N, N, _T_co]":
        """Evaluate the (matrix) exponential of this (square) matrix."""
        ...

    def mat_log[N: int](self: "Mat[N, N, _T_co]", /) -> "Mat[N, N, _T_co]":
        """Evaluate the (matrix) natural logarithm of this (square) matrix."""
        ...
