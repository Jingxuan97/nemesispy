#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
1D Atmospheric models
"""
import numpy as np
from scipy.integrate import solve_ivp
import scipy as sp
import matplotlib.pyplot as plt

# Define the unit dictionary
unit = {
    'pc': 3.08567e16,        # m parsec
    'ly': 9.460730e15,       # m lightyear
    'AU': 1.49598e11,        # m astronomical unit
    'R_sun': 6.95700e8,      # m solar radius
    'R_jup': 7.1492e7,       # m nominal equatorial Jupiter radius (1 bar pressure level)
    'R_e': 6.371e6,          # m nominal Earth radius
    'd_H2': 2.827e-10,       # m molecular diameter of H2
    'M_sun': 1.989e30,       # kg solar mass
    'M_jup': 1.8982e27,      # kg Jupiter mass
    'M_e': 5.972e24,         # kg Earth mass
    'amu': 1.66054e-27,      # kg atomic mass unit
    'atm': 101325,           # Pa atmospheric pressure
}

# Define the const dictionary
pi = np.pi
const = {
    'k_B': 1.38065e-23,         # J K-1 Boltzmann constant
    'sig_B': 5.67037e-8,        # W m-2 K-4 Stephan Boltzmann constant
    'R': 8.31446,               # J mol-1 K-1 universal gas constant
    'G': 6.67430e-11,           # m3 kg-1 s-2 universal gravitational constant
    'eps_LJ': 59.7*5.67037e-8,  # J depth of the Lennard-Jones potential well for H2
    'c_p': 14300,               # J K-1 hydrogen specific heat
    'h': 6.62607,               # Js Planck's constant
    'hbar': 1.05457,            # Js
    'N_A': 6.02214e23,          # Avagadro's number
}

class Model0:
    """
    Model 0: isothermal atmosphere at equilibrium temperature.

    Designed to be subclassed to create new model atmospheres.

    Assume:
        1)ideal gas law;
        2)hydrostatic equilibrium.

    Input:
        SMA: semimajor axis (m)
        P_range: an array of pressure coordinates, starting from highest pressure (Pascal)
        mmw: mean molecular weight (kg)

    All subclasses should have the following attributes:
        1)height grid (self.height());
        2)pressure grid (self.pressure());
        3)temperature grid (self.temperature()).
    """

    def __init__(self, T_star, R_star, M_plt, R_plt, SMA, P_range, mmw):
        # input parameters (all in SI units)
        self.T_star = T_star
        self.R_star = R_star
        self.M_plt = M_plt
        self.R_plt = R_plt
        self.SMA = SMA
        self.mmw = mmw
        self.P_range = P_range
        # derived parameters
        self.T_irr = self.T_star*(self.R_star/self.SMA)**0.5 # irradiation temperature
        self.T_eq = self.T_irr/2**0.5 # equilibrium temperature

        # make sure inputs are sensible
        assert min(self.T_star, self.R_star, self.M_plt, self.R_plt, self.SMA,
                   self.mmw, self.T_irr) > 0, "Inputs should be positive"
        assert min(P_range) > 0, "Inputs should be positive"
        assert max(P_range) < 1e8, "Maximum pressure shoul be less than 1000 bar"
        assert self.R_star < self.SMA, "R_star should be less than SMA"
        assert self.T_irr < self.T_star, "T_irr should be less than T_star"
        assert self.M_plt < unit['M_sun'], "M_plt shold be less than M_sun"
        assert self.R_plt < self.R_star, "R_plt should be less than R_star"
        assert self.mmw < 300*unit['amu'], "MMW should be less than 300 amu"

    def g(self, z):
        # calculate gravity as a function of z
        g = const['G']*self.M_plt/(self.R_plt+z)**2
        return g

    def rho(self, P, T):
        # calculate density as a function of P,T assuming ideal gas law
        rho = P*self.mmw/(const['k_B']*T)
        return rho

    def T_P_z_relation(self, P, z):
        # prescribe temperature as a function of P and z
        T = P*0+z*0 + self.T_eq
        return T

    def dz_dP(self, P, z):
        # calculate dz/dP assuming hydrostatic equilibrium
        T = self.T_P_z_relation(P, z)
        rho = self.rho(P, T)
        g = self.g(z)
        dzdP = -1/(rho*g)
        return dzdP

    def pressure(self):
        # output pressure grid, Pascal (same as input pressure grid)
        P = self.P_range
        return P

    def height(self):
        # solve the initial value problem dz/dP=-1/[rho(P,T(P,z))*g(z)], z[0]=P[0]
        # output height grid, m (same dimension as input pressure grid)
        sol = solve_ivp(fun=self.dz_dP,
                        t_span=[self.P_range[0], self.P_range[-1]],
                        y0=[0],
                        t_eval=self.P_range)
        H = sol.y[0]
        return H

    def temperature(self):
        # solve for temperature
        # output temperature grid, K (same dimension as input pressure grid)
        P = self.pressure()
        H = self.height()
        T = self.T_P_z_relation(P, H)
        return T


class Model1(Model0):
    """
    Model 1: PT profile following Guillot 2010.

    Input:
        k_th: mean thermal opacity (m2 kg-1)
        k_v: mean visible opacity (m2 kg-1)
        T_int: internal heat flux (K)
        f: heat redistribution efficiency
            (f = 1 at the substellar point, f = 1/2 for a day-side average,
            f = 1/4 for an averaging over the whole planetary surface)
    """
    def __init__(self, T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                     k_th, k_v, T_int=100, f=1):
        # initiate superclass Model0
        Model0.__init__(self, T_star, R_star, M_plt, R_plt, SMA, P_range, mmw)
        # extra model parameters
        self.k_th = k_th
        self.k_v = k_v
        self.T_int = T_int
        self.f = f

    def tau(self, P, z):
        # calculate optical depth as a function of P and z
        # defined as mean thermal opacity times column mass
        optical_depth = self.k_th * P/self.g(z)
        return optical_depth

    def T_P_z_relation(self, P, z):
        # calculate temperature as a function of pressure and altitude
        # temperature profile in terms of optical depth from eq29 of Guillot 2010
        T_int = self.T_int
        f = self.f
        gamma = self.k_v/self.k_th
        T_irr = self.T_irr
        t = self.tau(P, z)
        x = 3**0.5
        T = (0.75*T_int**4*(2/3+t)
             +0.75*T_irr**4*f*
             (2/3+1/(gamma*x)+(gamma/x-1/(gamma*x))*np.exp(-gamma*t*x)))**0.25

        # nemesis breaks if T > 10000
        if type(T) == int:
            if T > 5000.:
                T = 5000.
        else:
            for i in range(len(T)):
                if (T[i] > 5000.0):
                    T[i] = 5000.00
        return T


class Model2(Model0):
    """
    Model2: PT profile following Line et al 2013

    Input:
        kappa: thermal opacity
        gamma1: visible opacity 1
        gamma2: visible opacity 2
        alpha: percentage of gamma1 (or 2, check later)
        T_irr: irradiation temperature

    """
    def __init__(self, T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                     kappa, gamma1, gamma2, alpha, T_irr, T_int=100):
        Model0.__init__(self, T_star, R_star, M_plt, R_plt, SMA, P_range, mmw)

        # extra model parameters
        self.kappa = kappa
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.alpha = alpha
        self.T_irr = T_irr # irradiation temperature
        self.T_int = T_int

    def tau(self, P, z):
        # gray IR optical depth
        # calculate optical depth as a function of P and z
        # defined as mean thermal opacity times column mass
        optical_depth = self.kappa * P/self.g(z)
        return optical_depth

    def T_P_z_relation(self, P, z):
        # calculate temperature as a function of pressure and altitude
        # temperature profile from Line et al 2013
        T_int = self.T_int
        gamma1 = self.gamma1
        gamma2 = self.gamma2
        alpha = self.alpha
        T_irr = self.T_irr
        tau = self.tau(P, z)
        xi1 = ((2.0/3)*(1+(1/gamma1)*(1+(0.5*gamma1*tau-1)*np.exp(-gamma1*tau))
                +gamma1*(1-0.5*tau**2)*sp.special.expn(2, gamma1*tau)))
        xi2 = ((2.0/3)*(1+(1/gamma2)*(1+(0.5*gamma2*tau-1)*np.exp(-gamma2*tau))
                +gamma2*(1-0.5*tau**2)*sp.special.expn(2, gamma2*tau)))
        T = ( 0.75 *(T_int**4*(2.0/3.0+tau)
            +T_irr**4*(1-alpha)*xi1
            +T_irr**4*alpha*xi2) )**0.25

        # nemesis breaks if T > 10000
        if type(T) == int:
            if T > 5000.:
                T = 5000.
        else:
            for i in range(len(T)):
                if (T[i] > 5000.0):
                    T[i] = 5000.00
        return T



"""
### Model 0 Example: WASP 43b
T_star = 4520 # star temperature in K
R_star = 0.6668*unit['R_sun'] # star radius in m
M_plt = 2.052*unit['M_jup'] # planet mass in kg
R_plt = 1.036*unit['R_jup'] # planet radius in m
SMA = 0.015*unit['AU'] # Semimajor axis in m
NP = 100 # number of layers in the model atmosphere
P_top = 1e-5 # pressure at the top of the model atmosphere in bar
P_low = 1e1 # pressure at the bottom of the model atmosphere in bar
P_range = np.logspace(np.log10(P_low), np.log10(P_top), NP)*1e5 # pressure grid in Pascal
gas_mm = np.array([2.01588, 4.0])
gas_vmr = np.array([0.85, 0.15])
mmw = np.sum(gas_mm*gas_vmr*unit['amu'])
model0 = Model0(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw)
"""

"""
### Model 1 Example: WASP 43b
T_star = 4520 # star temperature in K
R_star = 0.6668*unit['R_sun'] # star radius in m
M_plt = 2.052*unit['M_jup'] # planet mass in kg
R_plt = 1.036*unit['R_jup'] # planet radius in m
SMA = 0.015*unit['AU'] # Semimajor axis in m
NP = 100 # number of layers in the model atmosphere
P_top = 1e-5 # pressure at the top of the model atmosphere in bar
P_low = 1e1 # pressure at the bottom of the model atmosphere in bar
P_range = np.logspace(np.log10(P_low), np.log10(P_top), NP)*1e5 # pressure grid in Pascal
# P_range = np.logspace(P_low, P_top, NP)*1e5 # test for error cathcing
gas_mm = np.array([2.01588, 4.0])
gas_vmr = np.array([0.85, 0.15])
mmw = np.sum(gas_mm*gas_vmr*unit['amu'])

k_th = 1e-3
k_v = 4e-4
T_int = 100
f = 1

model1 = Model1(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                k_th=k_th,
                k_v=k_v,
                T_int=T_int,
                f=f)

### Interactive plot of Model 1
# %matplotlib notebook
from matplotlib.widgets import Slider
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.30)

initial_kth = -3
initial_kv = -4
initial_Tint = 100
initial_f = 1

model1_plot  = Model1(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                    k_th=10**initial_kth,
                    k_v=10**initial_kv,
                    T_int=initial_Tint,
                    f=initial_f )


T = model1_plot.temperature()
P = model1_plot.pressure()*1e-5
l, = plt.plot(T, P, lw=2)
plt.grid()
plt.title('Model 1')
plt.ylabel('Pressure (bar)')
plt.xlabel('Temperature (K)')
plt.yscale('log')

ax = plt.axis([0, 5000, 1e-5/2, 1e1*2,])
plt.gca().invert_yaxis()

ax_kv = plt.axes([0.25, .12, 0.50, 0.02])
s_kv = Slider(ax_kv, r'log k$_v$', -5, 0, valinit=initial_kv)

ax_kth = plt.axes([0.25, .09, 0.50, 0.02])
s_kth = Slider(ax_kth, r'log k$_{th}$', -5, 0, valinit=initial_kth)

ax_Tint = plt.axes([0.25, .06, 0.50, 0.02])
s_Tint = Slider(ax_Tint, r'T$_{int}$', 0, 1000, valinit=initial_Tint)

ax_f = plt.axes([0.25, .03, 0.50, 0.02])
s_f = Slider(ax_f, r'$f$', 0, 1, valinit=initial_f)

def update(val):
    # read values from sliders
    k_v = s_kv.val
    k_th = s_kth.val
    T_int = s_Tint.val
    f = s_f.val
    # update curve
    model  = Model1(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                k_th=10**k_th,
                k_v=10**k_v,
                T_int=T_int,
                f=f)
    T =  model.temperature()
    l.set_xdata(T)
    # redraw canvas while idle
    fig.canvas.draw_idle()

s_kv.on_changed(update)
s_kth.on_changed(update)
s_Tint.on_changed(update)
s_f.on_changed(update)
plt.show()
"""

"""
### Model 2 Example: WASP 43b
T_star = 4520 # star temperature in K
R_star = 0.6668*unit['R_sun'] # star radius in m
M_plt = 2.052*unit['M_jup'] # planet mass in kg
R_plt = 1.036*unit['R_jup'] # planet radius in m
SMA = 0.015*unit['AU'] # Semimajor axis in m
NP = 100 # number of layers in the model atmosphere
P_top = 1e-5 # pressure at the top of the model atmosphere in bar
P_low = 1e1 # pressure at the bottom of the model atmosphere in bar
P_range = np.logspace(np.log10(P_low), np.log10(P_top), NP)*1e5 # pressure grid in Pascal
# P_range = np.logspace(P_low, P_top, NP)*1e5 # test for error cathcing
gas_mm = np.array([2.01588, 4.0])
gas_vmr = np.array([0.85, 0.15])
mmw = np.sum(gas_mm*gas_vmr*unit['amu'])

kappa = 1e-2
gamma1 = 1e-2
gamma2 = 1e-2
alpha = 0.5
T_irr = 2e3
T_int = 100

model2 = Model2(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                kappa = kappa,
                gamma1 = gamma1,
                gamma2 = gamma2,
                alpha = alpha,
                T_irr = T_irr,
                T_int = T_int)

### Interactive plot of Model 2
# %matplotlib notebook
from matplotlib.widgets import Slider
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.35)

initial_kappa = -2
initial_gamma1 = -2
initial_gamma2 = -2
initial_alpha = 0.5
initial_T_irr = 2e3
initial_NP = 100

model2_plot = Model2(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                kappa = 10**initial_kappa,
                gamma1 = 10**initial_gamma1,
                gamma2 = 10**initial_gamma2,
                alpha = initial_alpha,
                T_irr = initial_T_irr,
                T_int = 100)

T = model2_plot.temperature()
P = model2_plot.pressure()*1e-5
l, = plt.plot(T, P, lw=2)
plt.grid()
plt.title('Model 2')
plt.ylabel('Pressure (bar)')
plt.xlabel('Temperature (K)')
plt.yscale('log')

ax = plt.axis([0, 5000, 1e-5/2, 1e1*2,])
plt.gca().invert_yaxis()

ax_NP = plt.axes([0.25, .18, 0.50, 0.02])
s_NP = Slider(ax_NP, r'NP', 5, 100, valinit=initial_NP, valfmt='%1.f')

ax_kappa = plt.axes([0.25, .15, 0.50, 0.02])
s_kappa = Slider(ax_kappa, r'log kappa', -5, -2, valinit=initial_kappa)

ax_gamma1 = plt.axes([0.25, .12, 0.50, 0.02])
s_gamma1 = Slider(ax_gamma1, r'log gamma1', -3, 1, valinit=initial_gamma1)

ax_gamma2 = plt.axes([0.25, .09, 0.50, 0.02])
s_gamma2 = Slider(ax_gamma2, r'log gamma2', -3, 1, valinit=initial_gamma2)

ax_alpha = plt.axes([0.25, .06, 0.50, 0.02])
s_alpha = Slider(ax_alpha, r'alpha', 0, 1, valinit=initial_alpha)

ax_T_irr = plt.axes([0.25, .03, 0.50, 0.02])
s_T_irr = Slider(ax_T_irr, r'T$_{irr}$', 300, 3000, valinit=initial_T_irr)

def update(val):
    # amp is the current value of the slider
    NP = int(s_NP.val)
    kappa = s_kappa.val
    gamma1 = s_gamma1.val
    gamma2 = s_gamma2.val
    alpha = s_alpha.val
    T_irr = s_T_irr.val

    # update curve
    P_top = 1e-5 # pressure at the top of the model atmosphere in bar
    P_low = 1e1 # pressure at the bottom of the model atmosphere in bar
    P_range = np.logspace(np.log10(P_low), np.log10(P_top), NP)*1e5 # pressure grid in Pascal

    model  = Model2(T_star, R_star, M_plt, R_plt, SMA, P_range, mmw,
                kappa = 10**kappa,
                gamma1 = 10**gamma1,
                gamma2 = 10**gamma2,
                alpha = alpha,
                T_irr = T_irr,
                T_int = 100)

    T =  model.temperature()
    P =  model.pressure()*1e-5
    l.set_xdata(T)
    l.set_ydata(P)
    # redraw canvas while idle
    fig.canvas.draw_idle()

s_NP.on_changed(update)
s_kappa.on_changed(update)
s_gamma1.on_changed(update)
s_gamma2.on_changed(update)
s_alpha.on_changed(update)
s_T_irr.on_changed(update)
plt.show()
"""
