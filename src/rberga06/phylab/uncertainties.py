# pyright: reportUnknownMemberType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportOperatorIssue=false
from dataclasses import dataclass, field
from typing import Unpack, TypedDict, cast, final, override, TYPE_CHECKING
import sympy

if TYPE_CHECKING:
    class Expr(sympy.Expr):
        @override
        def __add__(self, other: sympy.Expr | complex | float | int) -> "Expr": ...  # type: ignore
        __rdiv__ = __div__ = __rmul__ = __mul__ = __rsub__ = __sub__ = __radd__ = __add__  # type: ignore

    class Symbol(sympy.Symbol):
        @override
        def __add__(self, other: sympy.Expr | complex | float | int) -> "Expr": ...  # type: ignore
        __rdiv__ = __div__ = __rmul__ = __mul__ = __rsub__ = __sub__ = __radd__ = __add__  # type: ignore
else:
    from sympy import Expr, Symbol


def expr(expr: sympy.Expr) -> Expr:
    return expr  # type: ignore


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


def symbol(name: str, **assumptions: Unpack[SympyAssumptions]) -> Symbol:
    return Symbol(name, **assumptions)

def symbols(*names: str, **assumptions: Unpack[SympyAssumptions]) -> tuple[Symbol, ...]:
    return tuple([symbol(name, **assumptions) for name in names])


@final
@dataclass
class Measure:
    best: float
    delta: float = 0

    def __pos__(self) -> "Measure": return self
    def __neg__(self) -> "Measure": return Measure(-self.best, self.delta)
    def __add__(self, other: float, /) -> "Measure": return Measure(self.best + other, self.delta)
    __radd__ = __add__
    def __sub__(self, other: float, /) -> "Measure": return Measure(self.best - other, self.delta)
    def __rsub__(self, other: float, /) -> "Measure": return Measure(other - self.best, self.delta)
    def __mul__(self, other: float, /) -> "Measure": return Measure(self.best * other, abs(self.delta * other))
    __rmul__ = __mul__
    def __div__(self, other: float, /) -> "Measure": return Measure(self.best / other, abs(self.delta / other))


@final
@dataclass(slots=True)
class Ctx:
    symbols: dict[Symbol, Measure] = field(default_factory=dict)

    def _delta(self, symbol: sympy.Symbol) -> Symbol:
        return Symbol(f"_pyphylab_delta_{symbol.name}", **symbol.assumptions0)

    def __setitem__(self, item: str | sympy.Symbol, value: Measure | tuple[float, float] | float):
        if isinstance(item, str):
            item = sympy.Symbol(item)
        if not isinstance(value, Measure):
            if isinstance(value, tuple):
                value = Measure(value[0], abs(value[1]))
            else:
                value = Measure(value, 0)
        self.symbols[cast(Symbol, item)] = value

    def __getitem__(self, item: str | sympy.Symbol) -> Measure:
        if isinstance(item, str):
            item = Symbol(item)
        return self.symbols[cast(Symbol, item)]

    def evalf(
        self, best_expr: sympy.Expr, /, *,
        n: int = 15, maxn: int = 100,
        subs: dict[sympy.Symbol, float],
        chop: bool | int = False, strict: bool = False, quad: str | None = None, verbose: bool = False,
        delta_upper_bound: bool = False,
    ) -> Measure:
        for x, v in self.symbols.items():
            dx = self._delta(x)
            # don't override user-selected substitutions
            if x not in subs:  subs[x] = v.best
            if dx not in subs: subs[dx] = v.delta
        # generate expression for delta
        if delta_upper_bound:
            delta_expr = sum([sympy.Abs(best_expr.diff(x)) * self._delta(x) for x in self.symbols], start=sympy.Integer(0))  # type: ignore
        else:
            delta_expr = sympy.sqrt(sum([(sympy.Abs(best_expr.diff(x)) * self._delta(x))**2 for x in self.symbols], start=sympy.Integer(0)))
        # perform evaluation
        best = best_expr.evalf(n=n, maxn=maxn, subs=subs, chop=chop, strict=strict, quad=quad, verbose=verbose)
        delta = expr(delta_expr).evalf(n=n, maxn=maxn, subs=subs, chop=chop, strict=strict, quad=quad, verbose=verbose)
        return Measure(best, delta)  # type: ignore
