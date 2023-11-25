#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false
# pyright: reportIncompatibleVariableOverride=false
"""Histogram for Distributions."""
from typing import Any, Sequence
from manim import BarChart
from manim.constants import MED_SMALL_BUFF
from manim.mobject.types.vectorized_mobject import VMobject, VGroup
from manim.mobject.text.tex_mobject import Tex
from manim.utils.color import ParsableManimColor, manim_colors, ManimColor
from ..measure import Measure, MeasureLike
from ..distribution import DiscreteDist


DEFAULT_BAR_COLORS = (
    manim_colors.BLUE_E,
    manim_colors.TEAL_E,
    manim_colors.GREEN_E,
    manim_colors.GOLD_E,
    manim_colors.YELLOW_E,
    manim_colors.MAROON_E,
    manim_colors.PURPLE_E,
)


class DiscreteDistributionHistogram[
    D: DiscreteDist[int] | DiscreteDist[Measure[int]] | DiscreteDist[MeasureLike[int]]
](BarChart):
    bar_labels: VGroup

    def __init__(
        self,
        dist: D,
        /, *,
        bar_names: Sequence[str] | None = None,
        bar_colors: Sequence[ManimColor | str] = DEFAULT_BAR_COLORS,
        **kwargs: Any,
    ) -> None:
        bins: list[float] = [len(b) for b in dist.bins]
        super().__init__(
            bins,
            bar_names=bar_names,
            bar_colors=bar_colors,  # type: ignore
            **kwargs,
        )

    def change_distribution(self, dist: D, /, *, update_colors: bool = True) -> None:
        bins: list[float] = [len(b) for b in dist.bins]
        return self.change_bar_values(bins, update_colors=update_colors)

    def add_bar_labels(self, /, *,
        color: ParsableManimColor | None = None,
        font_size: float = 24,
        buff: float = MED_SMALL_BUFF,
        label_constructor: type[VMobject] = Tex,
    ) -> VGroup:
        self.bar_labels = self.get_bar_labels(color, font_size, buff, label_constructor)
        self.add(self.bar_labels)
        return self.bar_labels
