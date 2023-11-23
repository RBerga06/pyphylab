#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for rberga06.phylab.poisson"""
from random import shuffle
from rberga06.phylab.poisson import Poisson

class TestPoisson:
    def test_data(self, /) -> None:
        data = [0]*12+[1]*10+[2]*7+[3]*5+[4]*1
        shuffle(data)
        p = Poisson(data)
        assert p.data is data
        assert len(p) == len(data)
