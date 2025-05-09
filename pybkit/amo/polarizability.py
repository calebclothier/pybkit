"""Helper functions for polarizability calculations."""

from __future__ import annotations
from typing import TYPE_CHECKING

import numpy as np
from scipy import constants
from sympy.physics.wigner import wigner_6j

if TYPE_CHECKING:
    from pybkit.amo.atom import AtomSpecies, FineEnergyLevel


# refer to this paper: https://www.epj.org/images/stories/news/2013/epj_d_01-05-13.pdf
# here omega refers to laser frequency


a0 = constants.physical_constants['Bohr radius'][0]


def transition_rate(
    atom: AtomSpecies,
    level1: FineEnergyLevel,
    level2: FineEnergyLevel
):
    """Computes transition rate between two fine levels."""
    omega12 = 2 * np.pi * atom.get_transition(level1, level2).frequency_Hz
    rme = atom.get_reduced_matrix_element(level1.uid, level2.uid)
    prefactor = 1 / (3 * np.pi * constants.epsilon_0 * constants.hbar * constants.c**3)
    rate = prefactor * omega12**3 * ((2 * level1.J + 1) / (2 * level2.J + 1)) * (rme * constants.e * a0)**2
    return rate


def polarizability(
    K: int,
    atom: AtomSpecies,
    level1: FineEnergyLevel,
    level2: FineEnergyLevel,
    omega: float, 
    part: str = 'real'
):
    """General expression for scalar/vector/tensor polarizability."""
    J1 = level1.J
    J2 = level2.J
    omega12 = 2 * np.pi * atom.get_transition(level1, level2).frequency_Hz
    rme = atom.get_reduced_matrix_element(level1.uid, level2.uid)
    prefactor = (-1)**(K + J1 + 1) * np.sqrt(2 * K + 1) * (-1)**J2 * wigner_6j(1, K, 1, J1, J2, J1)
    res = float(prefactor * (rme * constants.e * a0)**2 / constants.hbar)
    linewidth = transition_rate(atom, level1, level2)
    z = 1 / (omega12 - omega - 1j * linewidth / 2) + (-1)**K / (omega12 + omega + 1j * linewidth / 2)
    if part == 'real':
        res *= np.real(z)
    elif part == 'imag':
        res *= np.imag(z)
    else:
        raise ValueError('real or imag?')
    return res


def scalar_polarizability(
    atom: AtomSpecies,
    level1: FineEnergyLevel,
    level2: FineEnergyLevel,
    omega: float,
    part: str = 'real'
):
    """Scalar polarizabillity"""
    J = level1.J
    return 1 / np.sqrt(3 * (2 * J + 1)) * polarizability(0, atom, level1, level2, omega, part=part)


def vector_polarizability(
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
    prefactor = (-1)**(J + I + F) * np.sqrt(2 * F * (2 * F + 1) / (F + 1)) * float(wigner_6j(F, 1, F, J, I, J))
    return prefactor * polarizability(1, atom, level1, level2, omega, part=part)


def tensor_polarizability(
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
    prefactor = -(-1)**(J + I + F) * np.sqrt(2 * F * (2 * F - 1) * (2 * F + 1)/(3 * (F + 1) * (2 * F + 3))) * float(wigner_6j(F, 2, F, J, I, J))
    return prefactor * polarizability(2, atom, level1, level2, omega, part=part)


def get_3P1_core_polarizability(F: float, mF: float):
    if np.abs(mF) == F:
        core_polarizability = 7.27 + 18.01
    else:
        core_polarizability = 7.27 + 14.57
    return core_polarizability
