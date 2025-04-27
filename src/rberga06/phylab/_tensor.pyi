# pyright: reportExplicitAny = false
from collections.abc import Callable, Iterable, Iterator
from typing import Generic, TypeVar, TypeVarTuple, Any, Self, final, overload

type Elem = (
    Simd[Elem, *tuple[Any, ...]]
    | Mat[int, int, Elem]
    | Vec[int, Elem]
    | bool
    | int
    | float
)

_N_co = TypeVar("_N_co", bound=int, covariant=True)
_M_co = TypeVar("_M_co", bound=int, covariant=True)
_T_co = TypeVar("_T_co", bound=Elem, covariant=True)
_S = TypeVarTuple("_S")

@final
class Simd(Generic[_T_co, *_S]):
    """
    A n-dimensional array, where all operations happen element-wise
    (including matrix multiplication, reduction, etc.) and are parallelized.

    Some operations (`+`, `-`, `*`, `/`, `@`) also support an operand of the element type:
    in this case, the operation will happen, with that same operand, for all entries.
    """

    @staticmethod
    def from_it[X: Elem, N: int](
        _it: Iterable[X], _ty: type[N] = int, /
    ) -> "Simd[X, N]": ...
    @property
    def shape[S: tuple[int, ...]](self: "Simd[_T_co, *S]", /) -> S: ...
    # def __iter__[N: int](
    #     self: "Simd[_T_co, N, *_S]", /
    # ) -> Iterator["Simd[_T_co, *_S]"]: ...
    def __add__(self, rhs: Self | _T_co, /) -> Self: ...

    __radd__ = __pow__ = __truediv__ = __mul__ = __sub__ = __add__
    __rpow__ = __rsub__ = __rmul__ = __rtruediv__ = __radd__

    @overload
    def __matmul__[N: int, X: Elem](
        self: "Simd[Vec[N, X], *_S]", rhs: "Simd[Vec[N, X], *_S] | Vec[N, X]", /
    ) -> "Simd[X, *_S]": ...
    @overload
    def __matmul__[N: int, M: int, K: int, X: Elem](
        self: "Simd[Vec[N, X], *_S]", rhs: "Simd[Mat[N, M, X], *_S] | Mat[N, M, X]", /
    ) -> "Simd[Vec[M, X], *_S]": ...
    @overload
    def __matmul__[N: int, M: int, K: int, X: Elem](
        self: "Simd[Mat[N, M, X], *_S]",
        rhs: "Simd[Mat[M, K, X], *_S] | Mat[M, K, X]",
        /,
    ) -> "Simd[Mat[N, K, X], *_S]": ...
    @overload
    def __matmul__[N: int, M: int, X: Elem](
        self: "Simd[Mat[N, M, X], *_S]", rhs: "Simd[Vec[M, X], *_S] | Vec[M, X]", /
    ) -> "Simd[Vec[N, X], *_S]": ...
    @overload
    def __rmatmul__[N: int, X: Elem](
        self: "Simd[Vec[N, X], *_S]", lhs: "Vec[N, X]", /
    ) -> "Simd[X, *_S]": ...
    @overload
    def __rmatmul__[N: int, M: int, X: Elem](
        self: "Simd[Mat[N, M, X], *_S]", lhs: "Vec[N, X]", /
    ) -> "Simd[Vec[M, X], *_S]": ...
    @overload
    def __rmatmul__[N: int, M: int, K: int, X: Elem](
        self: "Simd[Mat[M, K, X], *_S]", lhs: "Mat[N, M, X]", /
    ) -> "Simd[Mat[N, K, X], *_S]": ...
    @overload
    def __rmatmul__[N: int, M: int, K: int, X: Elem](
        self: "Simd[Vec[M, X], *_S]", lhs: "Mat[N, M, X]", /
    ) -> "Simd[Vec[N, X], *_S]": ...
    def simd_map[U: Elem](self, _f: Callable[[_T_co], U], /) -> "Simd[U, *_S]": ...
    @property
    def as_scalar(self: "Simd[_T_co]", /) -> "_T_co":
        """Reinterpret the only element as a scalar."""
        ...

    @property
    def as_vector[N: int](self: "Simd[_T_co, N]", /) -> "Vec[N, _T_co]":
        """Reinterpret the only axis as a vector."""
        ...

    @property
    def as_simd_vec[X: Elem, N: int](
        self: "Simd[X, *_S, N]", /
    ) -> "Simd[Vec[N, X], *_S]":
        """Reinterpret the last axis as a vector."""
        ...

    @property
    def as_vec_simd[X: Elem, N: int](
        self: "Simd[X, N, *_S]", /
    ) -> "Vec[N, Simd[X, *_S]]":
        """Reinterpret the first axis as a vector."""
        ...

    @property
    def as_matrix[N: int, M: int](
        self: "Simd[_T_co, N, M]", /
    ) -> "Mat[N, M, _T_co]": ...
    @property
    def simd_join[X: Elem, *Z](
        self: "Simd[Simd[X, *Z], *_S]", /
    ) -> "Simd[X, *_S, *Z]": ...
    @property
    def mat_T[N: int, M: int, X: Elem](
        self: "Simd[Mat[N, M, X], *_S]", /
    ) -> "Simd[Mat[M, N, X], *_S]":
        """Transpose all elements, treating each of them as a matrix."""
        ...

    def mat_rk[N: int, M: int, X: Elem](
        self: "Simd[Mat[N, M, X], *_S]", /
    ) -> "Simd[int, *_S]": ...
    def mat_det[N: int, X: Elem](
        self: "Simd[Mat[N, N, X], *_S]", /
    ) -> "Simd[X, *_S]": ...
    def mat_inv[N: int, X: Elem](
        self: "Simd[Mat[N, N, X], *_S]", /
    ) -> "Simd[Mat[N, N, X], *_S]": ...
    def mat_exp[N: int, X: Elem](
        self: "Simd[Mat[N, N, X], *_S]", /
    ) -> "Simd[Mat[N, N, X], *_S]": ...
    def mat_log[N: int, X: Elem](
        self: "Simd[Mat[N, N, X], *_S]", /
    ) -> "Simd[Mat[N, N, X], *_S]": ...

@final
class Vec(Generic[_N_co, _T_co]):
    """A 1D vector."""

    def __init__(
        self, it: Iterable[_T_co], N_type: type[_N_co] | None = None, /
    ) -> None: ...
    @property
    def shape(self, /) -> tuple[_N_co]: ...
    def __add__(self, rhs: Self | _T_co, /) -> Self: ...

    __radd__ = __pow__ = __truediv__ = __mul__ = __sub__ = __add__
    __rpow__ = __rsub__ = __rmul__ = __rtruediv__ = __radd__

    def __matmul__(self, rhs: Self, /) -> _T_co:
        """This is just the standard dot product of two vectors."""
        ...

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

    @property
    def into_simd(self, /) -> "Simd[_T_co, _N_co, _M_co]":
        """Reinterpret `self` as a simd"""
        ...

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
