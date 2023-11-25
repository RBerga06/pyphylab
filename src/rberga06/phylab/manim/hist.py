#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false
# pyright: reportIncompatibleVariableOverride=false
"""Histogram for Distributions."""
from typing import Any, Sequence
from manim import BarChart, Circle
from manim.typing import Point3D
from manim.constants import MED_SMALL_BUFF, DEFAULT_DOT_RADIUS
from manim.mobject.types.vectorized_mobject import VMobject, VGroup
from manim.mobject.text.tex_mobject import Tex
from manim.mobject.geometry.line import DashedLine
from manim.utils.color import ParsableManimColor, manim_colors, ManimColor
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


class DiscreteDistributionHistogram[D: DiscreteDist[Any]](BarChart):
    y_range: tuple[float, float, float]
    dist: D
    bar_labels: VGroup
    avg_line: DashedLine
    expected_dots: VGroup

    def __init__(
        self,
        dist: D,
        /, *,
        bar_names: Sequence[str] | None = None,
        bar_colors: Sequence[ManimColor | str] = DEFAULT_BAR_COLORS,
        **kwargs: Any,
    ) -> None:
        self.dist = dist
        bins: list[float] = [len(b) for b in dist.bins]
        super().__init__(
            bins,
            bar_names=bar_names,
            bar_colors=bar_colors,  # type: ignore
            **kwargs,
        )

    def pt(self, x: float, y: float, /) -> Point3D:
        """Get the correct coordinates for a point in the graph."""
        return self.coords_to_point(x+.5, y, 0)  # type: ignore

    def add_bar_labels(
        self,
        /, *,
        color: ParsableManimColor | None = None,
        font_size: float = 24,
        buff: float = MED_SMALL_BUFF,
        label_constructor: type[VMobject] = Tex,
    ) -> VGroup:
        self.bar_labels = self.get_bar_labels(color, font_size, buff, label_constructor)
        self.add(self.bar_labels)
        return self.bar_labels

    def add_avg_line(self, /) -> DashedLine:
        self.avg_line = DashedLine(
            self.pt(self.dist.average, 0),
            self.pt(self.dist.average, self.y_range[1]),
        )
        self.add(self.avg_line)
        return self.avg_line

    def add_expected_dots(self, /) -> VGroup:
        self.expected_dots = VGroup(*[
            Circle(DEFAULT_DOT_RADIUS).move_to(self.pt(i, h))
            for i, h in enumerate(self.dist.expected())
        ])
        self.add(self.expected_dots)
        return self.expected_dots
