from __future__ import annotations
from typing import TYPE_CHECKING, Union
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.integrate import odeint
from scipy.integrate import solve_ivp

from ..amo.laser import GaussianLaser
from ..amo.atom import AtomSpecies, FineEnergyLevel, HyperfineEnergyLevel
from .. import units
from .geometry import Vector3D

if TYPE_CHECKING:
    from .generator import AODDevice, SLMDevice


class Atom:
    """Represents an individual atom that has been trapped by a tweezer."""

    def __init__(
        self,
        uid: int,
        species: AtomSpecies,
        position: np.ndarray = None,
        velocity: np.ndarray = None,
        tweezer: Tweezer = None
    ):
        self.uid = uid  # Unique identifier for this particular atom
        self.species = species  # Reference to the AtomSpecies for this atom
        self.position = position
        self.velocity = velocity
        self.tweezer = tweezer

    def __repr__(self):
        return f"Atom uid={self.uid} species={self.species.name}"
    
    @property
    def is_trapped(self):
        return self.tweezer is not None


class Tweezer:

    def __init__(self, position: Vector3D, laser: GaussianLaser):
        self.position = position
        self.atom = None
        self.laser = laser
        
    @property
    def has_atom(self):
        return self.atom is not None

    def beam_intensity_at_atom(self):
        if self.has_atom:
            z = r = 0  # Assume atom at beam focus, not exactly correct due to scattering force?
            return self.laser.intensity(z, r)
        else:
            raise RuntimeError('Tweezer does not have a trapped Atom')
    
    def trap_potential(
        self, 
        atom_species: AtomSpecies, 
        level: HyperfineEnergyLevel, 
        x: float,
        y: float,
        z: float,
        dz_x: float = 0, # astigmatism
        dz_y: float = 0, # astigmatism
    ):
        polarizability = atom_species.calculate_polarizability(
            level=level, wavelength=self.laser.wavelength, vector_coeff=0, unit='si')
        intensity = self.laser.intensity(x, y, z, dz_x, dz_y)
        U_trap =  -1 / (4 * constants.epsilon_0 * constants.c) * polarizability * intensity
        return U_trap
    
    def trap_force(
        self, 
        atom_species: AtomSpecies, 
        level: HyperfineEnergyLevel, 
        x: float,
        y: float,
        z: float,
        dz_x: float = 0, # astigmatism
        dz_y: float = 0, # astigmatism
    ): 
        polarizability = atom_species.calculate_polarizability(
            level=level, wavelength=self.laser.wavelength, vector_coeff=0, unit='si')
        intensity_gradient = self.laser.intensity_gradient(x, y, z, dz_x, dz_y)
        F_trap = 1 / (4 * constants.epsilon_0 * constants.c) * polarizability * intensity_gradient
        return F_trap
    
    def trap_depth(
        self,
        atom_species: AtomSpecies,
        level: HyperfineEnergyLevel,
        unit: str = 'J',
    ):
        depth = float(np.abs(self.trap_potential(atom_species, level, 0, 0, 0)))
        if unit in units.energy_si:
            return depth / units.energy_si[unit]
        elif unit in units.frequency_si:
            return depth / constants.hbar / units.frequency_si[unit]
        elif unit in units.temperature_si:
            return depth / constants.Boltzmann / units.temperature_si[unit]
        else:
            raise ValueError(f'Invalid unit "{unit}" for trap depth')
    
    def radial_trap_frequency(
        self,
        atom_species: AtomSpecies,
        level: HyperfineEnergyLevel,
        unit: str = 'Hz'
    ):
        U = self.trap_depth(atom_species, level, unit='J')
        wr = np.sqrt(4 * U / (atom_species.mass * self.laser.w0**2))
        if unit in units.frequency_si:
            return wr / units.frequency_si[unit]
        else:
            raise ValueError(f'Invalid frequency unit "{unit}"')
    
    def axial_trap_frequency(
        self,
        atom_species: AtomSpecies,
        level: HyperfineEnergyLevel,
        unit: str = 'Hz'
    ):   
        U = self.trap_depth(atom_species, level, unit='J')
        wz = np.sqrt(2 * U * self.laser.wavelength**2 / (np.pi**2 * atom_species.mass * self.laser.w0**4))
        if unit in units.frequency_si:
            return wz / units.frequency_si[unit]
        else:
            raise ValueError(f'Invalid frequency unit "{unit}"')
    
    def load_atom(self, atom: Atom):
        if atom.is_trapped:
            raise RuntimeError('Atom is already trapped in a Tweezer, must remove before loading into different Tweezer.')
        elif not self.has_atom:
            self.atom = atom
            atom.tweezer = self
        else:
            raise RuntimeError('Tweezer already has an Atom, must remove atom before loading a new one.')

    def remove_atom(self):
        if self.has_atom:
            self.atom.tweezer = None
            self.atom = None
        else:
            raise RuntimeError('Tweezer does not have an atom to remove')

    @property
    def has_atom(self):
        return self.atom is not None

    def __repr__(self):
        return f"Tweezer(position={self.position}, laser={self.laser}, atom={self.atom})"


class TweezerGroup:

    def __init__(
        self,
        tweezers: list[Tweezer],
        generator: Union[AODDevice, SLMDevice] = None
    ) -> None:
        self.tweezers = tweezers
        self.generator = generator
        
    def plot(self):
        plt.figure()
        for tweezer in self.tweezers:
            plt.scatter(tweezer.position.x, tweezer.position.y, c='C0')
        plt.xlabel('x position [$\mu$m]')
        plt.ylabel('y position [$\mu$m]')
        
        

class TweezerEnsemble:

    def __init__(self, pos=None, velocity=None, **auto_sample_kwargs):
        if pos is None and velocity is None:
            self.pos, self.velocity = maxwell_boltzmann_sample(**auto_sample_kwargs)
        else:
            if np.shape(pos)[0] != 3 or np.shape(velocity)[0] != 3:
                raise RuntimeError('The shape of pos and velocity must be (3, X)!')
            if np.shape(pos)[1] != np.shape(velocity)[1]:
                raise RuntimeError('The size of pos and velocity doesn\'t match!')
            self.pos = np.array(pos)
            self.velocity = np.array(velocity)
        self.ensemble_size = np.shape(self.pos)[1]

    def calc_energy_spec(self, energy_func, t=None, f_energy=1 / constants.Planck):
        if t is not None:
            V_xyz = lambda x, y, z: energy_func(t, x, y, z)
        else:
            V_xyz = energy_func

        kinetic_energy = 1 / 2 * mass * np.sum(self.velocity ** 2, axis=0)
        potential_energy = V_xyz(*self.pos)

        return (kinetic_energy + potential_energy) * f_energy