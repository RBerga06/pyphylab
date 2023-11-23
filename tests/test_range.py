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

    def test_ops(self, /) -> None:
        assert NotImplemented
