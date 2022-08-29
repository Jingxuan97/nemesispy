# -*- coding: utf-8 -*-
from scipy.special import erf
from scipy.special import voigt_profile
import numpy as np

def constant(x, C):
    return C * x/x

def lorentz(x, A, mu, gamma):
    return A * (gamma**2/( (x-mu)**2 + gamma**2))

def lorentz_plus_C(x, A, mu, gamma, C):
    out = lorentz(x, A, mu, gamma) + C
    return out

def normal(x, A, mu, sigma):
    out = A * 1/(sigma*(2*np.pi)**0.5) * np.exp(-0.5*(x-mu)**2/sigma**2)
    return out

def normal_plus_C(x, A, mu, sigma, C):
    out = normal(x, A, mu, sigma) + C
    return out

def skew_normal(x, A, mu, sigma, s):
    phi =  normal(x, A, mu, sigma)
    PHI = 0.5 * (1 + erf( s*((x-mu)/sigma) ))
    out = phi * PHI
    return out

def voigt(x, A, mu, sigma, gamma):
    return A*voigt_profile(x-mu,sigma,gamma)

def voigt_plus_C(x, A, mu, sigma, gamma, C):
    out = voigt(x, A, mu, sigma, gamma) + C
    return out

def skew_voigt(x, A, mu, sigma, gamma, skew):
    v = voigt(x, A, mu, sigma, gamma)
    PHI = 0.5 * (1 + erf( skew*((x-mu)/sigma) ))
    out = v * PHI
    return out

def fourier1(x, c0, c1, s1):
    x = x/180*np.pi
    sin = s1 * np.sin(x)
    cos = c1 * np.cos(x)
    return cos + sin + c0

def fourier2(x, c0, c1, c2, s1, s2):
    x = x/180*np.pi
    cos = c1 * np.cos(x) + c2 * np.cos(2*x)
    sin = s1 * np.sin(x) + s2 * np.sin(2*x)
    return cos + sin + c0

def fourier3(x, c0, c1, c2, c3, s1, s2, s3):
    x = x/180*np.pi
    cos = c1 * np.cos(x) + c2 * np.cos(2*x) + c3 * np.cos(3*x)
    sin = s1 * np.sin(x) + s2 * np.sin(2*x) + s3 * np.sin(3*x)
    return cos + sin + c0

def fourier4(x, c0, c1, c2, c3, c4, s1, s2, s3, s4):
    x = x/180*np.pi
    cos = c1 * np.cos(x) + c2 * np.cos(2*x) + c3 * np.cos(3*x) + c4 * np.cos(4*x)
    sin = s1 * np.sin(x) + s2 * np.sin(2*x) + s3 * np.sin(3*x) + s4 * np.sin(4*x)
    return cos + sin + c0
