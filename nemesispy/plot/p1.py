#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
"""
Template for making
"""
import matplotlib.pyplot as plt

nrows=2
ncols=3
fig, axs = plt.subplots(
    nrows=nrows,ncols=ncols,
    sharex=True,sharey=True,
    figsize=[8,4],
    dpi=600
    )
