"""Classes for representating laser configurations."""

from __future__ import annotations
from typing import Optional, Union

import numpy as np
from scipy import constants


class Laser:
    """Representation of a laser.

    Attributes:
        wavelength: [m]
        frequency: [Hz]
        linewidth: [Hz]
        power: [W]
    """

    def __init__(self, wavelength: float, linewidth: float, power: float):
        self.wavelength = wavelength
        self.frequency = constants.c / self.wavelength
        self.linewidth = linewidth
        self.power = power
        
    def __repr__(self) -> str:
        return f'Laser(wavelength={self.wavelength}, linewidth={self.linewidth}, power={self.power})'
    

class LaserPolarization:
    def __init__(self, pi: float, sigma_plus: float, sigma_minus: float):
        """
        Initialize the polarization components.
            pi: Component of π polarization
            sigma_plus: Component of σ⁺ polarization
            sigma_minus: Component of σ⁻ polarization
        """
        if (pi + sigma_plus + sigma_minus) != 1:
            raise ValueError('Polarization components must sum to 1')
        self.pi = pi
        self.sigma_plus = sigma_plus
        self.sigma_minus = sigma_minus


class GaussianLaser(Laser):
    """Representation of a Gaussian mode laser.

    Attributes:
        wavelength: [m]
        frequency: [Hz]
        linewidth: [Hz]
        power: [W]
        w0: beam waist [m]
        M: quality factor
        rayleigh_range: [m]
    """

    def __init__(
        self,
        wavelength: float,
        linewidth: float,
        power: float,
        w0: float,
        M: Optional[float] = 1
    ):
        super().__init__(wavelength, linewidth, power)
        self.w0 = w0
        self.M = M
        self.zR = np.pi * self.w0**2 / self.wavelength

    def beam_waist(self, z: float):
        """Calculates beam waist as a function of axial position z."""
        return self.w0 * np.sqrt(1 + (z / self.zR)**2)

    def intensity(
        self, 
        x: float, 
        y: float, 
        z: float, 
        dz_x: float = 0,
        dz_y: float = 0
    ) -> float:
        """Calculates beam intensity as a function of position relative to center.
        
        The arguments dz_x and dz_y help account for astigmatism in the trap during movement.
        """
        I0 = 2 * self.power / (np.pi * self.w0**2)
        qx = np.sqrt(1 + (z - dz_x) ** 2 / self.zR ** 2)
        qy = np.sqrt(1 + (z - dz_y) ** 2 / self.zR ** 2)
        expx = np.exp(-2 * x ** 2 / self.w0 ** 2 / (1 + (z - dz_x) ** 2 / self.zR ** 2))
        expy = np.exp(-2 * y ** 2 / self.w0 ** 2 / (1 + (z - dz_y) ** 2 / self.zR ** 2))
        I = I0 * expx * expy / qx / qy
        return I
    
    def intensity_gradient(
        self, 
        x: float, 
        y: float, 
        z: float, 
        dz_x: float = 0,
        dz_y: float = 0
    ) -> float:
        """Calculates beam intensity gradient as a function of position relative to center.
        
        The arguments dz_x and dz_y help account for astigmatism in the trap during movement.
        """
        I = self.intensity(x, y, z, dz_x, dz_y)
        qx = np.sqrt(1 + (z - dz_x) ** 2 / self.zR ** 2)
        qy = np.sqrt(1 + (z - dz_y) ** 2 / self.zR ** 2)
        dqxdz = 1 / qx * (z - dz_x) / self.zR ** 2
        dqydz = 1 / qy * (z - dz_y) / self.zR ** 2
        dIdx = - I * 4 * x / self.w0 ** 2 / qx ** 2
        dIdy = - I * 4 * y / self.w0 ** 2 / qy ** 2
        dIdz = I * (-1 / qx + 4 * x ** 2 / self.w0 ** 2 / qx ** 3) * dqxdz + \
               I * (-1 / qy + 4 * y ** 2 / self.w0 ** 2 / qy ** 3) * dqydz
        return np.array([dIdx, dIdy, dIdz])


def wavelength_to_rgb(wavelength, gamma=0.8):
    """
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm
    (789 THz through 400 THz).
    Based on code by Dan Bruton
    http://www.physics.sfasu.edu/astro/color/spectra.html
    """

    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R,G,B)