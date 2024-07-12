# pyright: reportUnknownMemberType=false
from dataclasses import dataclass, field
from typing import Unpack, TypedDict, NamedTuple, final
import sympy


class SympyAssumptions(TypedDict):
    pass


def symbol(name: str, **assumptions: Unpack[SympyAssumptions]) -> sympy.Symbol:
    return sympy.Symbol(name, **assumptions)

def symbols(*names: str, **assumptions: Unpack[SympyAssumptions]) -> tuple[sympy.Symbol, ...]:
    return tuple([symbol(name, **assumptions) for name in names])


@final
class Measure(NamedTuple):
    best: float
    delta: float


@final
@dataclass(slots=True)
class Ctx:
    symbols: dict[sympy.Symbol, Measure] = field(default_factory=dict)

    def _delta(self, symbol: sympy.Symbol) -> sympy.Symbol:
        return sympy.Symbol(f"_pyphylab_delta_{symbol.name}", **symbol.assumptions0)

    def __setitem__(self, item: str | sympy.Symbol, value: Measure | tuple[float, float] | float):
        if isinstance(item, str):
            item = sympy.Symbol(item)
        if not isinstance(value, Measure):
            if isinstance(value, tuple):
                value = Measure(value[0], abs(value[1]))
            else:
                value = Measure(value, 0)
        self.symbols[item] = value

    def __getitem__(self, item: str | sympy.Symbol) -> Measure:
        if isinstance(item, str):
            item = sympy.Symbol(item)
        return self.symbols[item]

    def evalf(
        self, expr: sympy.Expr, /, *,
        n: int = 15, maxn: int = 100,
        subs: dict[sympy.Symbol, float],
        chop: bool | int = False, strict: bool = False, quad: str | None = None, verbose: bool = False,
        delta_upper_bound: bool = False,
    ) -> Measure:
        for x, (v, dv) in self.symbols.items():
            dx = self._delta(x)
            # don't override user-selected substitutions
            if x not in subs:  subs[x] = v
            if dx not in subs: subs[dx] = dv
        # generate expression for delta
        if delta_upper_bound:
            delta_expr = sum([sympy.Abs(expr.diff(x)) * self._delta(x) for x in self.symbols], start=sympy.Integer(0))  # type: ignore
        else:
            delta_expr = sympy.sqrt(sum([(sympy.Abs(expr.diff(x)) * self._delta(x))**2], start=sympy.Integer(0)))  # type: ignore
        # perform evaluation
        best = expr.evalf(n=n, maxn=maxn, subs=subs, chop=chop, strict=strict, quad=quad, verbose=verbose)
        delta = delta_expr.evalf(n=n, maxn=maxn, subs=subs, chop=chop, strict=strict, quad=quad, verbose=verbose)  # type: ignore
        return Measure(best, delta)  # type: ignore
