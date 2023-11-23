#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for rberga06.phylab.measure"""
from rberga06.phylab.measure import Datum


class TestDatum:
    def test_ops(self, /) -> None:
        a = Datum(3, 1)
        b = Datum(-5, 2)
        c = Datum.from_delta_rel(3, .1)
        d = Datum.from_delta_rel(-5, .2)
        assert -a == Datum(-3, 1)
        assert -b == Datum(5, 2)
        assert a + b == Datum(3 - 5, 3)
        assert a - b == Datum(3 + 5, 3)
        assert c * d == Datum.from_delta_rel(-15, .3)
        assert c / d == Datum.from_delta_rel(-3/5,.3)
        assert a + 2 == Datum(3, 1)
        assert 2 + a == Datum(3, 1)
        assert b * -2 == Datum(10, 4)
        assert -2 * b == Datum(10, 4)
        assert b / -2 == Datum(-5/2, 1)
        assert -2 / d == Datum.from_delta_rel(-2/5, .2)

    def test_ε(self, /) -> None:
        assert Datum(0,.1).ε(0) == 0
        assert Datum(0,.1).ε(.05) == .5
        assert Datum(0,.1).ε(.1) == 1
        assert Datum(0,.1).ε(.2) == 2
