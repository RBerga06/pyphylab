#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportMissingTypeStubs=false
"""Manim CE support/utilities."""
from typing import Any, Sequence
from manim.utils.color import manim_colors, ManimColor
from manim import BarChart

from ..measure import MeasureLike
from ..distribution import DiscreteDist, Dist


DEFAULT_BAR_COLORS = (
    manim_colors.BLUE_E,
    manim_colors.TEAL_E,
    manim_colors.GREEN_E,
    manim_colors.GOLD_E,
    manim_colors.YELLOW_E,
    manim_colors.MAROON_E,
    manim_colors.PURPLE_E,
)


def hist_discrete[M: MeasureLike[int]](
    dist: DiscreteDist[M],
    /, *,
    bar_names: Sequence[str] | None = None,
    bar_colors: Sequence[ManimColor | str] = DEFAULT_BAR_COLORS,
    **kwargs: Any,
) -> BarChart:
    bins: list[float] = [len(b) for b in dist.bins]
    return BarChart(
        bins,
        bar_names=bar_names,
        bar_colors=bar_colors,  # type: ignore
        **kwargs,
    )


def hist[X: (int, float)](dist: Dist[X], /, **kwargs: Any) -> BarChart:
    raise NotImplementedError
