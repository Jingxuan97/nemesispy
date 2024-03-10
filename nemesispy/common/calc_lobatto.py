#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate quadrature points and weights for Gauss-Lobatto rules.
For disc integration, the range of cos(emission angle) is [0,1], and we choose
to not have quadurature point at 0.
See https://mathworld.wolfram.com/LobattoQuadrature.html
"""
from scipy.special import eval_legendre,legendre
import numpy as np
from functools import partial

def bisect(func, lower, upper, tol=1e-15):
    """
    Find root of function by bisection.

    Parameters
    ----------
    func : function
        Some pre-defined function.
    lower : real
        Lower bound of the root.
    upper : real
        Upper bound of the root.
    tol : real
        Error tolerance. Defaults to 1e-15.

    Returns
    -------
    root : real
        Root of the function between lower bound and upper bound.
    """
    assert func(lower) * func(upper) < 0
    while upper-lower > tol:
        median = lower + (upper-lower)/2
        f_median = func(median)
        if func(lower) * f_median < 0:
            upper = median
        else:
            lower = median
    root = median
    return root

def calc_legendre_derivative(n, x):
    """
    Calculate the derivative of degree n Legendre polynomial at x.

    Parameters
    ----------
    n : int
        Degree of the Legendre polynomia P_n(x).
    x : real
        Point at which to evaluate the derivative

    Returns
    -------
    result : real
        Derivative of P_n at x, i.e., P'_n(x).
    """
    result = (x*eval_legendre(n, x) - eval_legendre(n-1, x))\
        /((x**2-1)/n)
    return result


def lobatto(n):
    """
    Generate points and weights for Lobatto quadrature.

    Parameters
    ----------
    n : int
        Number of quadrature points apart from the end points -1 and 1.

    Returns
    -------
    points : ndarray
        Lobatto quadrature points.
    weights : ndarray
        Lobatto quadrature weights
    """
    assert n >= 1
    brackets = legendre(n-1).weights[:, 0]
    points = np.zeros(n)
    points[0] = -1
    points[-1] = 1
    for i in range(n-2):
        points[i+1] = bisect(
            partial(calc_legendre_derivative, n-1),
            brackets[i], brackets[i+1])
    points = np.around(points,decimals=14)
    weights = np.zeros(n)
    weight_end_pts = 2 / (n*(n-1))
    weights[0] = weight_end_pts
    weights[-1] = weight_end_pts
    for i in range(1,n-1):
        weights[i] = 2 / ( (n*(n-1)) * eval_legendre(n-1,points[i])**2 )
    return points, weights

def disc_weights(n):
    """
    Generate weights for disc integration in the the emission angle
    direction.

    Parameters
    ----------
    n : int
        Number of emission angles. Minuim 2.

    Returns:
    mu : ndarray
        List of cos(emission angle) for integration.
    wtmu : ndarray
        List of weights for integration.
    """
    assert n >= 2
    points, weights = lobatto(2*n)
    mu = np.zeros(n)
    wtmu = np.zeros(n)
    for i in range(n):
        mu[i] = abs(points[i])
        wtmu[i] = weights[i]
    mu = mu[::-1]
    wtmu = wtmu[::-1]
    return mu,wtmu