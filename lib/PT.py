"""
Contains pressure--temperature functions.
All functions must have pressure as the first parameter.

uniform: Isothermal profile.

adiabat: Adiabatic profile.

line: Parameterization from Line et al. 2013
xi: Helper function used when calculating the `line` profile.

spt: "Slope-PT" profile based on Piette & Madhusudhan 2020.
"""

import numpy as np
import scipy.constants as sc


def uniform(p, T0):
    """
    Returns a uniform temperature profile.
    """
    return np.zeros(p.size) + T0


def adiabat(p, gamma, T0, logp0):
     """
     Calculates an adiabatic temperature profile.
     """
     T = T0 / (1 + (gamma - 1) / gamma * np.log(10**logp0 / p))
     return T


def line(pressure, kappa,  gamma1, gamma2, alpha, beta, 
         R_star,   T_star, sma,    grav,   T_int=100, T_int_type='thorngren'):
  '''
  Generates a PT profile based on input free parameters and pressure array.
  If no inputs are provided, it will run in demo mode, using free
  parameters given by the Line 2013 paper and some dummy pressure
  parameters.

  Inputs
  ------
  pressure: 1D float ndarray
     Array of pressure values in bars.
  kappa : float, in log10. Planck thermal IR opacity in units cm^2/gr
  gamma1: float, in log10. Visible-to-thermal stream Planck mean opacity ratio.
  gamma2: float, in log10. Visible-to-thermal stream Planck mean opacity ratio.
  alpha : float.           Visible-stream partition (0.0--1.0).
  beta  : float.           A 'catch-all' for albedo, emissivity, and day-night
                           redistribution (on the order of unity)
  R_star: Float
     Stellar radius (in meters).
  T_star: Float
     Stellar effective temperature (in Kelvin degrees).
  sma:    Float
     Semi-major axis (in meters).
  grav:   Float
     Planetary surface gravity (at 1 bar) in cm/second^2.
  T_int:  Float
     Planetary internal heat flux (in Kelvin degrees).
     Default: 100
     Note: This parameter is ignored when `T_int_type` is set to 'thorngren'
  T_int_type: string.
     Method for determining `T_int`: 'const' (for a supplied constant value)
                                     'thorngren' (to use Thorngren et al. 2019)
     Default: thorngren

  Returns
  -------
  T: temperature array

  Example:
  --------
  >>> import PT as pt
  >>> import scipy.constants as sc
  >>> import matplotlib.pyplot as plt
  >>> import numpy as np

  >>> Rsun = 6.995e8 # Sun radius in meters

  >>> # Pressure array (bars):
  >>> p = np.logspace(2, -5, 100)

  >>> # Physical (fixed for each planet) parameters:
  >>> Ts = 5040.0        # K
  >>> Ti =  100.0        # K
  >>> a  = 0.031 * sc.au # m
  >>> Rs = 0.756 * Rsun  # m
  >>> g  = 2192.8        # cm s-2

  >>> # Fitting parameters:
  >>> kappa  = -1.5   # log10(3e-2)
  >>> gamma1 = -0.8   # log10(0.158)
  >>> gamma2 = -0.8   # log10(0.158)
  >>> alpha  = 0.5
  >>> beta   = 1.0
  >>> T0 = pt.PT(p, kappa, gamma1, gamma2, alpha, beta, Rs, Ts, Ti, a, g)

  >>> plt.figure(1)
  >>> plt.clf()
  >>> plt.semilogy(T0, p, lw=2, color="b")
  >>> plt.ylim(p[0], p[-1])
  >>> plt.xlim(800, 2000)
  >>> plt.xlabel("Temperature  (K)")
  >>> plt.ylabel("Pressure  (bars)")

  Developers:
  -----------
  Madison Stemm      astromaddie@gmail.com
  Patricio Cubillos  pcubillos@fulbrightmail.org

  Modification History:
  ---------------------
  2014-09-12  Madison   Initial version, adapted from equations (13)-(16)
                        in Line et al. (2013), Apj, 775, 137.
  2014-12-10  patricio  Reviewed and updated code.
  2015-01-22  patricio  Receive log10 of free parameters now.
  2019-02-13  mhimes    Replaced `params` arg with each parameter for 
                        consistency with other PT models
  2019-09-10  mhimes    Added T_int calculation from Thorngren et al. (2019)
  '''
  # Convert kappa, gamma1, gamma2 from log10
  kappa  = 10**(kappa )
  gamma1 = 10**(gamma1)
  gamma2 = 10**(gamma2)

  if T_int_type == 'thorngren':
      # Planetary internal temperature (Thorngren et al. 2019)
      # Hard-coded values are fitted parameters!
      T_eq  = (R_star/(2.0*sma))**0.5 * T_star
      F     = 4.0 * sc.Stefan_Boltzmann * T_eq**4
      T_int = 1.24 * T_eq * np.exp(-(np.log(F) - 0.14)**2 / 2.96)

  # Stellar input temperature (at top of atmosphere):
  T_irr = beta * (R_star / (2.0*sma))**0.5 * T_star

  # Gray IR optical depth:
  tau = kappa * (pressure*1e6) / grav # Convert bars to barye (CGS)

  xi1 = xi(gamma1, tau)
  xi2 = xi(gamma2, tau)

  # Temperature profile (Eq. 13 of Line et al. 2013):
  temperature = (0.75 * (T_int**4 * (2.0/3.0 + tau) +
                         T_irr**4 * (1-alpha) * xi1 +
                         T_irr**4 * alpha     * xi2 ) )**0.25

  return temperature


def xi(gamma, tau):
  """
  Calculate Equation (14) of Line et al. (2013) Apj 775, 137

  Parameters:
  -----------
  gamma: Float
     Visible-to-thermal stream Planck mean opacity ratio.
  tau: 1D float ndarray
     Gray IR optical depth.

  Modification History:
  ---------------------
  2014-12-10  patricio  Initial implemetation.
  """
  return (2.0/3) * \
         (1 + (1./gamma) * (1 + (0.5*gamma*tau-1)*np.exp(-gamma*tau)) +
          gamma*(1 - 0.5*tau**2) * sp.expn(2, gamma*tau)              )


def spt(p, T0, dTbot_32, dT32_10, dT10_0, dT0_1, dT1_01, dT01_001, dT001_top):
    '''
    Non-inverted atmospheric profile based on the "SPT" model of 
    Piette & Madhusudhan (2020), modified to work on arbitrary pressure grids.

    Inputs
    ------
    p: 1D float ndarray
        Atmospheric pressure profile (in bar).
    T0: float
        Temperature at pressure layer closest to 3.2 bar. Reference temperature 
        for the other params.
    dTbot_32: float
        Temperature difference between the bottom of the atmosphere and the 
        pressure layer closest to 32 bar
    dT32_10: float
        Temperature difference between the pressure layer closest to 32 bar and 
        the pressure layer closest to 10 bar
    dT10_0: float
        Temperature difference between the pressure layer closest to 10 bar and 
        the pressure layer closest to 3.2 bar
    dT0_1: float
        Temperature difference between the pressure layer closest to 3.2 bar and
        the pressure layer closest to 1 bar
    dT1_01: float
        Temperature difference between the pressure layer closest to 1 bar and 
        the pressure layer closest to 0.1 bar
    dT01_001: float
        Temperature difference between the pressure layer closest to 100 mbar 
        and the pressure layer closest to 10 mbar
    dT001_top: float
        Temperature difference between the layer closest to 10 mbar and the 
        top of the atmosphere
    '''
    T = np.zeros(p.shape)
    # Indices of pressure layers
    ilaytop = np.argmin(p)
    ilay001 = np.argmin(np.abs(p - 0.01))
    ilay01  = np.argmin(np.abs(p - 0.1))
    ilay1   = np.argmin(np.abs(p - 1))
    ilay0   = np.argmin(np.abs(p - 3.2))
    ilay10  = np.argmin(np.abs(p - 10))
    ilay32  = np.argmin(np.abs(p - 32))
    ilaybot = np.argmax(p)
    # Temperatures at those layers
    T[ilay0]   = T0
    T[ilay10]  = T0         + dT10_0
    T[ilay32]  = T[ilay10]  + dT32_10
    T[ilaybot] = T[ilay32]  + dTbot_32
    T[ilay1]   = T0         - dT0_1
    T[ilay01]  = T[ilay1]   - dT1_01
    T[ilay001] = T[ilay01]  - dT01_001
    T[ilaytop] = T[ilay001] - dT001_top
    # Linear spline interpolate
    ilays = np.array([ilaytop, ilay001, ilay01, ilay1, ilay0, ilay10, ilay32, ilaybot])
    rep = si.splrep(np.log10(p[ilays]), T[ilays], k=1)
    T   = si.splev(np.log10(p), rep)
    # Gaussian smoothing - sigma is 0.3 dex, Piette & Madhusudhan (2020)
    sig = 0.3 / np.abs(np.log10(p)[0] - np.log10(p)[1])
    return gaussian_filter1d(T, sigma=sig, mode='nearest')


