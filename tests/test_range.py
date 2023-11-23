#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for rberga06.phylab.range"""
from rberga06.phylab.range import R, Range

class TestRange:
    def test_mk(self, /) -> None:
        r = Range("(", 3, 6.9, "]")
        assert Range.mk("(3;6.9]") == r
        assert r == r[:] == r[:12] == r[-1:]
        assert R[-1:.2] == Range("[", -1, .2, "]")

    def test_split(self, /) -> None:
        assert R[0:5].split(5) == (
            Range("[", 0, 1, "]"),
            Range("(", 1, 2, "]"),
            Range("(", 2, 3, "]"),
            Range("(", 3, 4, "]"),
            Range("(", 4, 5, "]"),
        )

    def test_contains(self, /) -> None:
        assert 0 not in R[1:]
        assert 1 not in R[:-1]
        assert .5 in R[0:]
        assert .5 in R[:1]
        assert 567438 in R
        assert 0 not in R["(0;+1]"]
        assert 0 not in R["[-1;0)"]
        assert 0 in R["[0;+1)"]
        assert 0 in R["(-1;0]"]
