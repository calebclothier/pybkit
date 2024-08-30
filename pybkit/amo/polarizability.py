"""Helper functions for polarizability calculations."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from scipy import constants
from sympy.physics.wigner import wigner_6j

if TYPE_CHECKING:
    from .atom import AtomSpecies, FineEnergyLevel


# refer to this paper: https://www.epj.org/images/stories/news/2013/epj_d_01-05-13.pdf
# here omega refers to laser frequency


a0 = constants.physical_constants['Bohr radius'][0]


def gamma(
    atom: AtomSpecies,
    level1: FineEnergyLevel,
    level2: FineEnergyLevel
):
    """Computes transition linewidth for two fine levels."""
    J1 = level1.J
    E12 = atom.get_transition(level1, level2).energy_J
    omega12 = E12 / constants.hbar
    rme = atom.get_reduced_matrix_element(level1.uid, level2.uid)
    res = (omega12**3) / (3 * np.pi * constants.epsilon_0 * constants.Planck * constants.c**3)
    res *= (1 / (2 * J1 + 1)) * (rme * constants.e * a0)**2
    return res


def alpha_K(
    K: int,
    atom: AtomSpecies,
    level1: FineEnergyLevel,
    level2: FineEnergyLevel,
    omega: float, part='real'
):
    """General expression for scalar/vector/tensor polarizability."""
    J0 = level1.J
    J1 = level2.J
    E12 = atom.get_transition(level1, level2).energy_J
    omega12 = E12 / constants.hbar
    rme = atom.get_reduced_matrix_element(level1.uid, level2.uid)
    prefactor = (-1)**(K+J0+1) * np.sqrt(2*K+1) * (-1)**J1 * wigner_6j(1,K,1,J0,J1,J0)
    res = float(prefactor * (rme * constants.e * a0)**2 / constants.hbar)
    linewidth = gamma(atom, level1, level2)
    z = 1/(omega12 - omega - 1j*linewidth/2) + (-1)**K/(omega12 + omega + 1j*linewidth/2)
    if part == 'real':
        res *= np.real(z)
    elif part == 'imag':
        res *= np.imag(z)
    else:
        raise ValueError('real or imag?')
    return res


def alpha_S(
    atom: AtomSpecies,
    level1: FineEnergyLevel,
    level2: FineEnergyLevel,
    omega: float,
    part: str = 'real'
):
    """Scalar polarizabillity"""
    J = level1.J
    return 1/np.sqrt(3*(2*J + 1))*alpha_K(0, atom, level1, level2, omega, part=part)


def alpha_V(
    atom: AtomSpecies,
    level1: FineEnergyLevel,
    level2: FineEnergyLevel,
    omega: float,
    F: float,
    part: str = 'real'
):
    """Vector polarizability"""
    I = atom.I
    J = level1.J
    prefactor = (-1)**(J + I + F) * np.sqrt(2*F*(2*F+1)/(F+1)) * float(wigner_6j(F, 1, F, J, I, J))
    return prefactor * alpha_K(1, atom, level1, level2, omega, part=part)


def alpha_T(
    atom: AtomSpecies,
    level1: FineEnergyLevel,
    level2: FineEnergyLevel,
    omega: float,
    F: float,
    part: str = 'real'
):
    """Tensor polarizability"""
    I = atom.I
    J = level1.J
    six_j = wigner_6j(F, 2, F,J, I, J)
    prefactor = -(-1)**(J + I + F) * np.sqrt(2*F*(2*F-1)*(2*F+1)/(3*(F+1)*(2*F+3))) * six_j
    return prefactor * alpha_K(2, atom, level1, level2, omega, part=part)


def get_3P1_core_polarizability(F: float, mF: float):
    if np.abs(mF) == F:
        core_polarizability = 7.27 + 18.01
    else:
        core_polarizability = 7.27 + 14.57
    return core_polarizability
