#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for rberga06.phylab.measure"""
from rberga06.phylab.measure import AnyMeasure, Datum

EQ_THRESHOLD = 1e-20


class TestDatum:
    def eq(self, x: AnyMeasure, y: AnyMeasure, /) -> None:
        assert (x.delta - y.delta) <= EQ_THRESHOLD
        assert (x - y).best <= EQ_THRESHOLD

    def test_ops(self, /) -> None:
        a = Datum(3, 1)
        b = Datum(-5, 2)
        c = Datum.from_delta_rel(3, .1)
        d = Datum.from_delta_rel(-5, .2)
        self.eq(  -a,    Datum(-3, 1))
        self.eq(  -b,    Datum(5, 2))
        self.eq( a + b,  Datum(3 - 5, 3))
        self.eq( a - b,  Datum(3 + 5, 3))
        self.eq( c * d,  Datum.from_delta_rel(-15, .3))
        self.eq( c / d,  Datum.from_delta_rel(-3/5,.3))
        self.eq( a + 2,  Datum(5, 1))
        self.eq( 2 + a,  Datum(5, 1))
        self.eq( b * -2, Datum(10, 4))
        self.eq( -2 * b, Datum(10, 4))
        self.eq( b / -2, Datum(5/2, 1))
        self.eq( -2 / d, Datum.from_delta_rel(2/5, .2))

    def test_ε(self, /) -> None:
        assert Datum(0,.1).ε(0) == 0
        assert Datum(.05,.1).ε(0) == .5
        assert Datum(.1,.1).ε(0) == 1
        assert Datum(.2,.1).ε(0) == 2
