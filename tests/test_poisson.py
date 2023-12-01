#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for rberga06.phylab.poisson"""
from random import shuffle
from rberga06.phylab.dataset import DataSet
from rberga06.phylab.poisson import Poisson

class TestPoisson:
    def test_data(self, /) -> None:
        # Create a dataset
        data = [0]*12+[1]*10+[2]*7+[3]*5+[4]*1
        shuffle(data)
        ds = DataSet(data)
        bins = ds.intbins()
        # Test the Poisson fit
        fit = Poisson.fit(bins)
        assert fit.data.orig.data is data
        assert fit.dist.average == fit.data.average
