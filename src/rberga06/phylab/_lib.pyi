from typing import final, overload
import numpy as np
from numpy.typing import NDArray

@final
class uf64:
    best: float
    delta: float
    delta_rel: float
    @overload
    def __init__(self, best: float = 0.0, /, delta: float = 0.0) -> None: ...
    @overload
    def __init__(self, best: float = 0.0, /, *, delta_rel: float) -> None: ...

@final
class uvf64:
    best: NDArray[np.float64]
    delta: NDArray[np.float64] | float
    delta_rel: NDArray[np.float64] | float
    @overload
    def __init__(self, best: float = 0.0, /, delta: float = 0.0) -> None: ...
    @overload
    def __init__(self, best: float = 0.0, /, *, delta_rel: float) -> None: ...

__all__ = ["uf64", "uvf64"]
