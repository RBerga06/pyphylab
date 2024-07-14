# pyright: reportIncompatibleMethodOverride=false
# pyright: reportUnknownMemberType=false
from collections.abc import Sequence, Mapping
from typing import Any, TypedDict, Unpack, cast
from sympy import Expr, Symbol, Rational
from .measure import Measure, Datum


class SympyAssumptions(TypedDict, total=False):
    commutative: bool
    infinite: bool
    finite: bool
    hermitian: bool
    antihermitian: bool
    complex: bool
    algebraic: bool
    transcendental: bool
    extended_real: bool
    real: bool
    imaginary: bool
    rational: bool
    irrational: bool
    integer: bool
    noninteger: bool
    even: bool
    odd: bool
    prime: bool
    composite: bool
    zero: bool
    nonzero: bool
    extended_nonzero: bool
    positive: bool
    nonnegative: bool
    negative: bool
    nonpositive: bool
    extended_positive: bool
    extended_nonnegative: bool
    extended_negative: bool
    extended_nonpositive: bool


class MeasureSymbol(Symbol):
    measure: Measure[float] | None = None

    def measure_delta_symbol(self) -> Symbol:
        return Symbol(f"delta[{self.name}]", **cast(SympyAssumptions, self.assumptions0))


def symbol(name: str, value: None | float | tuple[float, float] | Measure[float] = None, /, **assumptions: Unpack[SympyAssumptions]) -> MeasureSymbol:
    # TODO: Improve assumptions based on value
    x = MeasureSymbol(name, **assumptions)
    if value is None:
        return x
    if isinstance(value, float | int):
        x.measure = Datum(value, 0)
    elif isinstance(value, tuple):
        x.measure = Datum(*value)
    else:
        x.measure = value
    return x


def symbols(names: Sequence[str], values: Sequence[None | float | tuple[float, float] | Measure[float]] | None = None, **assumptions: Unpack[SympyAssumptions]) -> tuple[MeasureSymbol, ...]:
    if values is None:
        return tuple([symbol(n, **assumptions) for n in names])
    symbols: tuple[MeasureSymbol, ...] = ()
    for name, value in zip(names, values):
        symbols += (symbol(name, value, **assumptions), )
    return symbols


class _EvalfKwArgs(TypedDict, total=False):
    n: int
    maxn: int
    subs: Mapping[Symbol, float | Measure[float]]
    chop: bool | int
    strict: bool
    quad: str | None
    verbose: bool


def _evalf(expr: Expr, subs: Mapping[Symbol, float | Measure[float]], kwargs: _EvalfKwArgs, /) -> Any:
    kwargs["subs"] = {**subs, **kwargs.get("subs", {})}
    return expr.evalf(**kwargs)


def evalf(expr: Expr, /, use_delta_upper_bound: bool = False, **kwargs: Unpack[_EvalfKwArgs]) -> Measure[float]:
    symbols = expr.atoms(Symbol)
    measure_subs = {x: x.measure for x in symbols if isinstance(x, MeasureSymbol) and x.measure is not None}
    if use_delta_upper_bound:
        return _evalf(expr, measure_subs, kwargs)  # type: ignore
    else:
        best: float = _evalf(expr, {x: m.best for x, m in measure_subs.items()}, kwargs)
        delta: float = _evalf(
            sum([abs(expr.diff(x))**2 * abs(x.measure_delta_symbol()) for x in measure_subs]) ** Rational(1, 2),  # type: ignore
            {x.measure_delta_symbol(): m.delta for x, m in measure_subs.items()}, kwargs
        )
        return Datum(best, delta)


__all__ = ["symbol", "symbols", "evalf"]
