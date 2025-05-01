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
        object.__setattr__(self, "_n", self._np.shape[0])
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

    def __iter__(self, /) -> Iterator[_T_co]:
        yield from self._np


@final
@dataclass(frozen=True, slots=True)
class Mat(Generic[_N_co, _M_co, _T_co]):
    """A 2D matrix (N rows, M columns)."""

    _n: _N_co
    _m: _M_co
    _np: np.ndarray[tuple[_N_co, _M_co], np.dtype[np.object_]]

    def __init__(
        self,
        it: Iterable[Iterable[_T_co]],
        n: _N_co | None = None,
        m: _M_co | None = None,
        /,
    ) -> None:
        object.__setattr__(self, "_np", np.array(it))
        object.__setattr__(self, "_n", self._np.shape[0])
        object.__setattr__(self, "_m", self._np.shape[1])
        if n is not None:
            assert self._n == n

    @property
    def shape(self, /) -> tuple[_N_co, _M_co]:
        return (self._n, self._m)

    def __iter__(self, /) -> Iterator[Vec[_M_co, _T_co]]:
        for row in self._np:
            yield Vec(row, self._m)

    def __add__(self, rhs: Self | _T_co, /) -> Self:
        if isinstance(rhs, Mat):
            return type(self)(self._np + rhs._np, self._n, self._m)
        return type(self)(self._np + rhs, self._n, self._m)

    def __sub__(self, rhs: Self | _T_co, /) -> Self:
        if isinstance(rhs, Mat):
            return type(self)(self._np - rhs._np, self._n, self._m)
        return type(self)(self._np - rhs, self._n, self._m)

    def __rsub__(self, lhs: Self | _T_co, /) -> Self:
        if isinstance(lhs, Mat):
            return type(self)(lhs._np - self._np, self._n, self._m)
        return type(self)(lhs - self._np, self._n, self._m)

    def __mul__(self, rhs: Self | _T_co, /) -> Self:
        if isinstance(rhs, Mat):
            return type(self)(self._np * rhs._np, self._n, self._m)
        return type(self)(self._np * rhs, self._n, self._m)

    def __truediv__(self, rhs: Self | _T_co, /) -> Self:
        if isinstance(rhs, Mat):
            return type(self)(self._np / rhs._np, self._n, self._m)
        return type(self)(self._np / rhs, self._n, self._m)

    def __rtruediv__(self, lhs: Self | _T_co, /) -> Self:
        if isinstance(lhs, Mat):
            return type(self)(lhs._np / self._np, self._n, self._m)
        return type(self)(lhs / self._np, self._n, self._m)

    def __pow__(self, rhs: Self | _T_co, /) -> Self:
        if isinstance(rhs, Mat):
            return type(self)(self._np**rhs._np, self._n, self._m)
        return type(self)(self._np**rhs, self._n, self._m)

    def __rpow__(self, lhs: Self | _T_co, /) -> Self:
        if isinstance(lhs, Mat):
            return type(self)(lhs._np**self._np, self._n, self._m)
        return type(self)(lhs**self._np, self._n, self._m)

    __radd__ = __add__
    __rmul__ = __mul__

    @overload
    def __matmul__[K: int](
        self, rhs: "Mat[_M_co, K, _T_co]", /
    ) -> "Mat[_N_co, K, _T_co]": ...
    @overload
    def __matmul__(self, rhs: Vec[_M_co, _T_co], /) -> Vec[_N_co, _T_co]: ...
    def __matmul__[K: int](
        self, rhs: "Mat[_M_co, K, _T_co] | Vec[_M_co, _T_co]", /
    ) -> "Mat[_N_co, K, _T_co] | Vec[_N_co, _T_co]":
        if isinstance(rhs, Mat):
            return Mat(self._np @ rhs._np, self._n, rhs._m)
        else:
            return Vec(self._np @ rhs._np, self._n)

    def __rmatmul__(self, lhs: Vec[_N_co, _T_co], /) -> Vec[_M_co, _T_co]:
        return Vec(lhs._np @ self._np, self._m)

    @property
    def T(self, /) -> "Mat[_M_co, _N_co, _T_co]":
        """Transpose this matrix."""
        return Mat(self._np.T, self._m, self._n)

    def diag(self: "Mat[_N_co, _N_co, _T_co]", /) -> "Vec[_N_co, _T_co]":
        return Vec(np.diag(self._np), self._n)

    # @property
    # def into_vec[X: Elem](
    #     self: "Mat[_N_co, _M_co, X]", /
    # ) -> "Vec[_N_co, Vec[_M_co, X]]":
    #     """Reinterpret `self` as a vector of vectors (rows)"""
    #     ...

    # @property
    # def into_simd(self, /) -> "Simd[_T_co, _N_co, _M_co]":
    #     """Reinterpret `self` as a simd"""
    #     ...

    def rk(self, /) -> int:
        """Evaluate the rank of this matrix."""
        return np.linalg.matrix_rank(self._np)  # pyright: ignore[reportArgumentType]

    def det[N: int](self: "Mat[N, N, _T_co]", /) -> _T_co:
        """Evaluate the determinant of this matrix."""
        return np.linalg.det(self._np)  # pyright: ignore[reportArgumentType]

    def inv[N: int](self: "Mat[N, N, _T_co]", /) -> "Mat[N, N, _T_co]":
        """Evaluate the inverse of this matrix."""
        return Mat(np.linalg.inv(self._np), self._n, self._n)  # pyright: ignore[reportCallIssue, reportUnknownArgumentType, reportArgumentType]

    # def mat_exp[N: int](self: "Mat[N, N, _T_co]", /) -> "Mat[N, N, _T_co]":
    #     """Evaluate the (matrix) exponential of this (square) matrix."""
    #     raise NotImplementedError

    # def mat_log[N: int](self: "Mat[N, N, _T_co]", /) -> "Mat[N, N, _T_co]":
    #     """Evaluate the (matrix) natural logarithm of this (square) matrix."""
    #     ...
