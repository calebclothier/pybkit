"""Classes for representing atomic energy levels and performing useful computations."""

from __future__ import annotations
from typing import Optional, List, Tuple, Union
from abc import ABC
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants

from .laser import GaussianLaser, wavelength_to_rgb
from .polarizability import alpha_S, alpha_V, alpha_T, get_3P1_core_polarizability
from .. import units


Hz_to_cm = 3.335641e-11
cm_to_J = 1.9864e-23
a0 = constants.physical_constants['Bohr radius'][0]
term_symbol_L_map = {
    0: 'S',
    1: 'P',
    2: 'D',
    3: 'F',
    4: 'G',
    5: 'H',
    6: 'I'}


class EnergyLevel(ABC):
    """Abstract base class for all energy level classes."""

    def __init__(self, energy) -> None:
        self.energy = energy


class FineEnergyLevel(EnergyLevel):
    """Representation of a fine structure energy level of an atom.

    Attributes:
        config (str): Electron configuration, e.g., '6s6p'
        energy (float): Energy level [cm^-1] (relative to ground state)
        S (float): Total electronic spin
        L (float): Total orbital angular momentum
        J (float): Total angular momentum
        uid (str): Unique identifier for the energy level
        atom (Atom): Parent atom
        lande_g (float): Landé g-factor
        core_polarizability (float): Core polarizability [atomic units]
        hyperfine_constant (float): Magnetic dipole hyperfine constant A [Hz]
    """

    def __init__(
        self,
        config: str,
        energy: float,
        S: float,
        L: float,
        J: float,
        uid: Optional[str] = None,
        atom: Optional['AtomSpecies'] = None,
        lande_g: Optional[float] = None,
        core_polarizability: Optional[float] = 0,
        hyperfine_constant: Optional[float] = 0
    ):
        self.config = config
        self.energy = energy
        self.S = S
        self.L = L
        self.J = J
        self._uid = uid
        self.atom = atom
        self.lande_g = lande_g
        self.core_polarizability = core_polarizability
        self.hyperfine_constant = hyperfine_constant
        self.validate()

    @property
    def term_symbol(self) -> str:
        """Generates and returns the term symbol for the energy level."""
        if self.S is not None and self.L is not None:
            return f'{2*self.S+1}{term_symbol_L_map[self.L]}{self.J}'
        else:
            return None

    @property
    def uid(self) -> str:
        """Returns the unique identifier or generates one if not specified."""
        if self._uid is not None:
            return self._uid
        else:
            return str(self)

    @property
    def formatted_label(self) -> str:
        """Returns a LaTeX-formatted label for the energy level."""
        s = self.config.replace('[Xe]', '')
        if '4f^{14}' in s:
            s = s.replace('4f^{14}', '')
        s = f"${s}$"
        if self.term_symbol:
            s += f' $^{2*self.S+1}{term_symbol_L_map[self.L]}_{self.J}$'
        else:
            s += f' $J={self.J}$'
        return s

    def __str__(self) -> str:
        if self.term_symbol:
            return f'{self.config} {self.term_symbol}'
        else:
            return f'{self.config} J={self.J}'

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(uid='{self.uid}')"

    def __eq__(self, __value: object) -> bool:
        if isinstance(__value, (FineEnergyLevel, HyperfineEnergyLevel)):
            return __value.uid == self.uid
        else:
            return False

    def validate(self):
        """Validates the provided J value against S and L according to the triangle rule."""
        assert self.J is not None, "Must provide a value for J"
        if self.S is not None and self.L is not None:
            min_J, max_J = abs(self.S - self.L), abs(self.S + self.L)
            assert min_J <= self.J <= max_J, 'Invalid value for J'

    def calculate_hyperfine_shift(
        self,
        I: float,
        F: float
    ) -> float:
        """Calculates the hyperfine energy shift for specified quantum numbers.

        Args:
            I: Total nuclear spin
            F: Total electronic + nuclear spin

        Returns:
            float: Hyperfine shift [Hz]
        """
        # units: Hz
        if not self.hyperfine_constant:
            return 0
        K = F * (F + 1) - I * (I + 1) - self.J * (self.J + 1)
        return 0.5 * K * self.hyperfine_constant

    def calculate_zeeman_shift(
        self,
        F: float,
        mF: float,
        B_field: float,
        I: Optional[float] = None,
        gI: Optional[float] = None
    ) -> float:
        """Calculates the Zeeman energy shift for specified quantum numbers and magnetic field.

        If the energy level is associated with an Atom, then the atomic values for I and gI will be
        used unless an alternative value is provided.

        Args:
            F: Total electronic + nuclear spin
            mF: Projection of total spin
            B_field: External magnetic field [T]
            I: Total nuclear spin
            gI: Nuclear gyromagnetic factor

        Returns:
            float: Zeeman shift [Hz]
        """
        mu_B = constants.physical_constants['Bohr magneton'][0]
        mu_N = constants.physical_constants['nuclear magneton'][0]
        if self.atom is None and (I is None or gI is None):
            raise ValueError('Must either specify a parent atom or both parameters I, gI')
        if I is None:
            I = self.atom.I
        if gI is None:
            gI = 2 * self.atom.nuclear_magneton * mu_N / mu_B
        if self.lande_g is None:
            gJ = 0
        else:
            gJ = self.lande_g
        gF = gJ + (gI - gJ) * (F*(F+1) + I*(I+1) - self.J*(self.J+1)) / (2*F*(F+1))
        dE = -mu_B * gF * mF * B_field / constants.Planck
        return dE

    def get_hyperfine_levels(
        self,
        B_field: float,
        I: Optional[float] = None
    ) -> List['HyperfineEnergyLevel']:
        """Generates a list of hyperfine levels based on the magnetic field and nuclear spin.

        Calculates the level splitting due to both hyperfine coupling and Zeeman effect.
        The uid for each `HyperfineEnergyLevel` is the id of the parent `FineEnergyLevel`
        plus the hyperfine quantum numbers "F=X mF=Y"

        Args:
            B_field: External magnetic field [T],
            I: Total nuclear spin

        Returns:
            List[HyperfineEnergyLevel]: list of hyperfine energy levels
        """
        if I is None and self.atom is None:
            raise ValueError("Must either specify a parent atom or the nuclear spin I")
        if I is None:
            I = self.atom.I
        min_F, max_F = abs(self.J - I), abs(self.J + I)
        n_levels = int(max_F - min_F) + 1
        levels = []
        for i in range(n_levels):
            F = min_F + i
            for j in range(int(2*F + 1)):
                mF = F - j
                hf_shift = self.calculate_hyperfine_shift(I=I, F=F)
                hf_shift *= Hz_to_cm
                zeeman_shift = self.calculate_zeeman_shift(I=I, F=F, mF=mF, B_field=B_field)
                zeeman_shift *= Hz_to_cm
                hf_energy = self.energy + hf_shift + zeeman_shift
                level = HyperfineEnergyLevel(
                    uid=f'{self.uid} F={F} mF={mF}',
                    fine_level=self,
                    config=self.config,
                    energy=hf_energy,
                    S=self.S,
                    L=self.L,
                    J=self.J,
                    I=I,
                    F=F,
                    mF=mF)
                levels.append(level)
        return levels


class HyperfineEnergyLevel(FineEnergyLevel):
    """Representation of a hyperfine structure energy level of an atom.

    Attributes:
        config (str): Electron configuration, e.g., '6s6p'
        energy (float): Energy level [cm^-1] (relative to ground state)
        S (float): Total electronic spin
        L (float): Total orbital angular momentum
        J (float): Total angular momentum
        I (float):
        F (float):
        mF (float):
        uid (str): Unique identifier for the energy level
        fine_level (FineEnergyLevel):
        atom (AtomSpecies): Parent atom
    """

    def __init__(
        self,
        config,
        energy,
        S, L, J,
        I, F, mF,
        uid=None,
        fine_level=None
    ):
        super().__init__(
            uid=uid,
            config=config,
            energy=energy,
            S=S, L=L, J=J)
        self.I = I
        self.F = F
        self.mF = mF
        self.fine_level = fine_level
        min_F, max_F = abs(self.J - self.I), abs(self.J + self.I)
        assert min_F <= self.F <= max_F, 'Invalid value for F'
        assert -self.F <= self.mF <= self.F, 'Invalid value for mF'

    def __str__(self) -> str:
        return super().__str__() + f' F={self.F} mF={self.mF}'

    @property
    def formatted_label(self) -> str:
        s = super().formatted_label
        s += f' $F={self.F}$ $m_F={self.mF}$'
        return s


class ImaginaryEnergyLevel(FineEnergyLevel):
    """Representation of an imaginary fine energy level of an atom (for calculations)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        
class EnergyTransition:

    def __init__(
        self, 
        level1: EnergyLevel,
        level2: EnergyLevel,
        detuning_Hz: float = 0. # sign matters
    ):
        if level1 == level2:
            raise ValueError(f'Cannot have transition between identical levels: {level1}')
        self.level1 = level1
        self.level2 = level2
        self.detuning_Hz = detuning_Hz
        self.energy_J = self.calculate_energy(unit='J')
        self.frequency_Hz = self.calculate_frequency(unit='Hz')
        self.wavelength_m = self.calculate_wavelength(unit='m')
        
    def __repr__(self) -> str:
        return f"EnergyTransition(level1='{self.level1.uid}', level2='{self.level2.uid}', detuning_Hz={self.detuning_Hz})"
    
    @property
    def color(self):
        wl_nm = self.calculate_wavelength('nm')
        if wl_nm < 380:
            return 'purple'
        elif wl_nm > 700:
            return 'brown'
        else:
            return wavelength_to_rgb(wl_nm)
        
    def calculate_detuning(self, unit: str = 'nm') -> float:
        """Calculate the detuning in specified units"""
        if unit in units.frequency_si:
            return self.detuning_Hz / units.frequency_si[unit]
        elif unit in units.length_si:
            dL = constants.c / (self.frequency_Hz + self.detuning_Hz) - constants.c / self.frequency_Hz
            return dL / units.length_si[unit]
        elif unit == 'J':
            return constants.h * self.detuning_Hz
        else:
            raise ValueError(f'Invalid detuning unit "{unit}"')
    
    def calculate_energy(self, unit: str = 'J') -> float:
        """Calculate transition energy in specified units"""
        dE = (self.level2.energy - self.level1.energy)
        if unit == 'J':
            return dE * cm_to_J + self.calculate_detuning('J')
        elif unit == 'cm^{-1}':
            return dE + self.calculate_detuning('J') / cm_to_J
        else:
            raise ValueError(f'Invalid energy unit "{unit}"')

    def calculate_frequency(self, unit: str = 'THz') -> float:
        """Calculate transition frequency [Hz] between two energy levels."""
        if unit not in units.frequency_si:
            raise ValueError(f'Invalid frequency unit "{unit}"') 
        dE = self.calculate_energy('J')
        f = dE / constants.h
        f /= units.frequency_si[unit]
        return f

    def calculate_wavelength(self, unit: str = 'nm') -> float:
        """Calculate transition wavelength [m] between two energy levels."""
        if unit not in units.length_si:
            raise ValueError(f'Invalid wavelength unit "{unit}"') 
        f = self.calculate_frequency('Hz')
        l = constants.c / f
        l /= units.length_si[unit]
        return np.abs(l)
    

class AtomSpecies:
    """Representation of an atomic species (i.e., specific isotope).
    
    Attributes:
        fine_levels (List[FineEnergyLevel]): Atomic fine structure
        I (float): Total nuclear spin
        nuclear_magneton (float): Nuclear magneton 
        B_field (float): External magnetic field [T]
    """

    def __init__(
        self,
        fine_levels: List['FineEnergyLevel'],
        nuclear_spin: float,
        nuclear_magneton: float,
        B_field: float
    ):
        self.I = nuclear_spin
        self.nuclear_magneton = nuclear_magneton
        self.B_field = B_field
        self._fine_level_dict = {level.uid: level for level in fine_levels}
        self._hyperfine_level_dict = {hf_level.uid: hf_level for level in fine_levels \
            for hf_level in level.get_hyperfine_levels(I=self.I, B_field=self.B_field)}
        self._reduced_matrix_elements = self.load_reduced_matrix_elements()

    @property
    def fine_levels(self) -> List['FineEnergyLevel']:
        """Get list of all fine energy levels for atom."""
        return list(self._fine_level_dict.values())

    @property
    def fine_level_ids(self) -> List[str]:
        """Get list of all fine energy level uids for atom."""
        return list(self._fine_level_dict.keys())

    @property
    def hyperfine_levels(self) -> List[HyperfineEnergyLevel]:
        """Get list of all hyperfine energy levels for atom."""
        return list(self._hyperfine_level_dict.values())

    @property
    def hyperfine_level_ids(self) -> List[str]:
        """Get list of all hyperfine energy level uids for atom."""
        return list(self._hyperfine_level_dict.keys())

    def get_fine_level(self, level: str) -> FineEnergyLevel:
        """Get fine energy level by uid."""
        return self._fine_level_dict[level]

    def get_hyperfine_level(self, level: str) -> HyperfineEnergyLevel:
        """Get hyperfine energy level by uid."""
        return self._hyperfine_level_dict[level]

    def get_level(self, level: str) -> EnergyLevel:
        """Get energy level by uid."""
        if isinstance(level, EnergyLevel):
            return level
        elif level in self.fine_level_ids:
            return self.get_fine_level(level)
        elif level in self.hyperfine_level_ids:
            return self.get_hyperfine_level(level)
        else:
            raise KeyError(f'Level "{level}" does not exist for Atom "{str(self)}"')
        
    def get_transition(
        self,
        level1: Union[str, EnergyLevel],
        level2: Union[str, EnergyLevel]
    ) -> EnergyTransition:
        if isinstance(level1, str):
            level1 = self.get_level(level1)
        if isinstance(level2, str):
            level2 = self.get_level(level2)
        return EnergyTransition(level1=level1, level2=level2)

    def get_nearby_transitions(
        self,
        wavelength: float,
        max_detuning: float,
        level: Union[str, EnergyLevel] = None,
        hyperfine: bool = True
    ) -> List[EnergyTransition]:
        """Get transitions within detuning of specified wavelength.
        
        Args:
            wavelength: Transition wavelength [m]
            max_detuning: Maximum detuning from transition wavelength [Hz]
            level: Initial or final energy level of transition,
            hyperfine: Search through hyperfine energy levels if True,
                otherwise will only search fine energy levels
            
        Returns:
            List[dict]: List of dictionaries specifying each transition
        """
        f0 = constants.c / wavelength
        transitions = []
        if level and not hyperfine:
            l1 = self.get_level(level)
            for l2 in self.fine_levels:
                if l1 == l2:
                    continue
                transition = self.get_transition(l1, l2)
                detuning = np.abs(f0 - transition.frequency_Hz)
                if detuning < max_detuning:
                    detuned_transition = EnergyTransition(
                        level1=l1, 
                        level2=l2,
                        detuning_Hz=detuning)
                    transitions.append(detuned_transition)
        elif level and hyperfine:
            l1 = self.get_level(level)
            for l2 in self.fine_levels:
                for hf_l2 in l2.get_hyperfine_levels(B_field=self.B_field):
                    if l1 == hf_l2:
                        continue
                    transition = self.get_transition(l1, hf_l2)
                    detuning = np.abs(f0 - transition.frequency_Hz)
                    if detuning < max_detuning:
                        detuned_transition = EnergyTransition(
                            level1=l1, 
                            level2=hf_l2,
                            detuning_Hz=detuning)
                        transitions.append(detuned_transition)
        elif not level and not hyperfine:
            for l1 in self.fine_levels:
                for l2 in self.fine_levels:
                    if l1 == l2:
                        continue
                    transition = self.get_transition(l1, l2)
                    detuning = np.abs(f0 - transition.frequency_Hz)
                    if detuning < max_detuning:
                        detuned_transition = EnergyTransition(
                            level1=l1, 
                            level2=l2,
                            detuning_Hz=detuning)
                        transitions.append(detuned_transition)
        return sorted(transitions, key=lambda transition: transition.detuning_Hz)

    def load_reduced_matrix_elements(self) -> dict:
        """Loads reduced matrix elements from file. Should be overridden in subclass."""
        return {}

    def get_reduced_matrix_element(self, level1: str, level2: str, source: bool = False):
        """Gets the reduced matrix element between two energy levels."""
        try:
            entry = self._reduced_matrix_elements[level1][level2]
            if source:
                return entry
            else:
                return entry['value']
        except KeyError:
            return None

    def calculate_polarizability(
        self,
        level: Union[str, 'HyperfineEnergyLevel'],
        wavelength: Union[float, np.ndarray],
        vector_coeff: float,
        unit: str = 'atomic'
    ) -> Union[float, np.ndarray]:
        """Calculates polarizability of a hyperfine energy level at given wavelength.
        
        Underlying theory: https://www.epj.org/images/stories/news/2013/epj_d_01-05-13.pdf
        
        Args:
            level: Hyperfine energy level
            wavelength: Light wavelength [m]
            vector_coeff: Coefficient for vector polarizability
        Returns:
            Union[float, np.ndarray]: Polarizability [atomic units]
        """
        hf_level = self.get_level(level)
        assert isinstance(hf_level, HyperfineEnergyLevel), 'Level must be a HyperfineEnergyLevel'
        fine_level1 = hf_level.fine_level
        assert fine_level1 is not None, 'HyperfineEnergyLevel must have a parent FineEnergyLevel'
        F = hf_level.F
        mF = hf_level.mF
        omega = 2*np.pi*constants.c/wavelength
        if isinstance(fine_level1.core_polarizability, (int, float, complex)):
            core_polarizability = fine_level1.core_polarizability
        elif callable(fine_level1.core_polarizability):
            core_polarizability = fine_level1.core_polarizability(F, mF)
        else:
            raise TypeError('Invalid type for core_polarizability')
        if F == 1/2:
            tensor_coeff = 0
        else:
            tensor_coeff = (3*mF**2 - F*(F + 1))/(F*(2*F - 1))
        mF_over_F = 0
        if F != 0:
            mF_over_F = mF/F
        prefactor = constants.m_e * constants.e**2 / (a0*constants.epsilon_0*constants.Planck)**2
        sum_S = 0
        sum_V = 0
        sum_T = 0
        for fine_level2 in self.fine_levels:
            if self.get_reduced_matrix_element(fine_level1.uid, fine_level2.uid):
                sum_S += alpha_S(self, fine_level1, fine_level2, omega)
                sum_V += alpha_V(self, fine_level1, fine_level2, omega, F)
                sum_T += alpha_T(self, fine_level1, fine_level2, omega, F)
        alpha_au = prefactor * (sum_S + vector_coeff*mF_over_F*sum_V + tensor_coeff*sum_T)/4
        alpha_au += core_polarizability
        if unit == 'atomic':
            return alpha_au
        elif unit == 'si':
            au_to_si = 4 * np.pi * constants.epsilon_0 * a0**3
            return alpha_au * au_to_si
        else:
            raise ValueError(f'Invalid unit type "{unit}", valid options are "atomic" or "si"')

    def calculate_scattering_rate(
        self,
        level: Union[str, 'HyperfineEnergyLevel'],
        wavelength: Union[float, np.ndarray],
        vector_coeff: float
    ) -> Union[float, np.ndarray]:
        """Calculates scattering rate of a hyperfine energy level at given wavelength.
        
        Underlying theory found here https://www.epj.org/images/stories/news/2013/epj_d_01-05-13.pdf
        
        Args:
            level: Hyperfine energy level
            wavelength: Light wavelength [m]
            vector_coeff: Coefficient for vector polarizability
        Returns:
            Union[float, np.ndarray]: Scattering rate [atomic units]
        """
        hf_level = self.get_level(level)
        assert isinstance(hf_level, HyperfineEnergyLevel), 'Level must be a HyperfineEnergyLevel'
        fine_level1 = hf_level.fine_level
        assert fine_level1 is not None, 'HyperfineEnergyLevel must have a parent FineEnergyLevel'
        F = hf_level.F
        mF = hf_level.mF

        omega = 2*np.pi*constants.c/wavelength
        if isinstance(fine_level1.core_polarizability, (int, float, complex)):
            core_polarizability = fine_level1.core_polarizability
        elif callable(fine_level1.core_polarizability):
            core_polarizability = fine_level1.core_polarizability(F, mF)
        else:
            raise TypeError('Invalid type for core_polarizability')
        mF_over_F = 0
        if F != 0:
            mF_over_F = mF/F
        prefactor = constants.m_e * constants.e**2 / (a0*constants.epsilon_0*constants.Planck)**2
        sum_S = 0
        sum_V = 0
        for fine_level2 in self.fine_levels:
            if self.get_reduced_matrix_element(fine_level1.uid, fine_level2.uid):
                sum_S += alpha_S(self, fine_level1, fine_level2, omega, part='imag')
                sum_V += alpha_V(self, fine_level1, fine_level2, omega, F, part='imag')
        return prefactor * (sum_S + vector_coeff*mF_over_F*sum_V)/4 #+ core_polarizability

    def calculate_light_shift(self, level, F, mF, laser):
        raise NotImplementedError

    def plot_fine_structure(
        self,
        transitions: list[EnergyTransition] = None,
        term_symbols_only: bool = True,
        unit: str = 'cm^{-1}'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plots all energy levels in atomic fine structure."""
        # validate units
        if unit == 'cm^{-1}':
            factor = 1
        elif unit in units.frequency_si:
            factor = cm_to_J / constants.Planck / units.frequency_si[unit]
        else:
            raise ValueError(f'Invalid unit "{unit}"')
        # helper for arranging according to values of S, L
        def arrange_level(level):
            if level.term_symbol:
                if level.S == 0:
                    x0 = 0
                elif level.S == 1:
                    x0 = 0.5
                x0 += level.L * 0.15
            else:
                x0 = 1.1
            return x0
        # plot levels
        fig, ax = plt.subplots(figsize=(8,8))
        config_colors = {}
        i = 0
        for level in self.fine_levels:
            if term_symbols_only and not level.term_symbol:
                continue
            if not level.term_symbol:
                color = 'gray'
            else:
                term_L = term_symbol_L_map[level.L]
                if term_L in config_colors:
                    color = config_colors[term_L]
                else:
                    color = f'C{i}'
                    config_colors[term_L] = color
                    i += 1
            x0 = arrange_level(level)
            ax.hlines(factor * level.energy, xmin=x0, xmax=x0+0.1, color='gray') #color)
            # ax.text(x=x0+0.1, y=factor * level.energy, s=level.formatted_label, color='gray')
            if level.S == 1 and level.J != (level.L - 1) and level.L != 0:
                ax.text(x=x0+0.1, y=factor * level.energy, s=level.J, fontsize=8, va='center', color='gray')
            else:
                if level.S == 1 and level.L > 0:
                    label = '_'.join([*level.formatted_label.split('_')[:-1], '{J}$'])
                    ax.text(x=x0+0.1, y=factor * level.energy, s=level.J, fontsize=8, va='center', color='gray')
                else:
                    label = level.formatted_label
                ax.text(x=x0, y=factor * level.energy, s=label, va='top', color='gray')
        # plot transitions
        if transitions:
            for transition in transitions:
                x = arrange_level(transition.level1)
                y = factor * transition.level1.energy
                xp = arrange_level(transition.level2)
                yp = factor * transition.level2.energy
                dx = xp - x
                dy = yp - y
                C='black'
                ax.arrow(x+0.05,y,dx,dy, color=transition.color, zorder=999)
                ax.text(x+dx/2+0.06, y+dy/2, f"{transition.calculate_wavelength('nm'):.1f} nm", 
                        fontsize=8, ha='left', va='center', color=transition.color)
        # # first ionization energy
        # ax.hlines(y=factor * 50443.20, xmin=0, xmax=1, color='black', linestyle='dashed')
        xmax = 1.05 if term_symbols_only else 1.5
        ax.set_xlim(-0.05, xmax)
        ax.set_ylabel(f'Energy [${unit}$]')
        ax.xaxis.set_visible(False)
        fig.tight_layout()
        return fig, ax

    def plot_hyperfine_structure(
        self,
        transitions: list[EnergyTransition] = None,
        term_symbols_only: bool = True,
        label: bool = True,
        unit: str = 'cm^{-1}'
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plots all energy levels in atomic hyperfine structure."""
        # validate units
        if unit == 'cm^{-1}':
            factor = 1
        elif unit in units.frequency_si:
            factor = cm_to_J / (constants.Planck * units.frequency_si[unit])
        else:
            raise ValueError(f'Invalid unit "{unit}"')
        # plot levels
        fig, ax = plt.subplots(figsize=(8,8))
        config_colors = {}
        i = 0
        for level in self.fine_levels:
            for hf_level in level.get_hyperfine_levels(B_field=self.B_field, I=self.I):
                if term_symbols_only and hf_level.term_symbol is None:
                    continue
                elif hf_level.term_symbol is None:
                    color = 'gray'
                else:
                    term_L = term_symbol_L_map[hf_level.L]
                    if term_L in config_colors:
                        color = config_colors[term_L]
                    else:
                        color = f'C{i}'
                        config_colors[term_L] = color
                        i += 1
                ax.hlines(factor*hf_level.energy, xmin=hf_level.mF-0.25, xmax=hf_level.mF+0.25, color=color)
                if label:
                    ax.text(x=hf_level.mF-0.25, y=factor*hf_level.energy, s=hf_level.formatted_label, color=color)
        # plot transitions
        if transitions:
            for transition in transitions:
                x = transition.level1.mF
                y = factor * transition.level1.energy
                xp = transition.level2.mF
                yp = factor * transition.level2.energy
                dx = xp - x
                dy = yp - y
                ax.arrow(x,y,dx,dy, color=transition.color, zorder=999, length_includes_head=True)
                         #width=2e-4*factor, head_width=0.05, head_length=5e2*factor,)
                # ax.annotate("", xy=(x,y), xytext=(dx,dy), arrowprops=dict(arrowstyle="->", color=transition.color))
                ax.text(x+dx/2+0.06, y+dy/2, f"{transition.calculate_wavelength('nm'):.3f} nm", 
                        fontsize=8, ha='left', va='center', color=transition.color,)
        # # first ionization energy
        # xmin, xmax = ax.get_xlim()
        # ax.hlines(y=factor * 50443.20, xmin=xmin, xmax=xmax, color='black', linestyle='dashed')
        ax.xaxis.set_visible(False)
        ax.set_ylabel(f'Energy [${unit}$]')
        fig.tight_layout()
        return fig, ax


class Yb171(AtomSpecies):
    """Representation of Yb-171 atomic isotope."""
    
    proton_number = 70
    mass = 170.936 * 1.66e-27
    nuclear_spin = 1/2
    nuclear_magneton = 0.4919
    
    def __init__(self, B_field: float):
        fine_energy_levels = self.load_fine_levels_from_json()
        fine_energy_levels.append(
            ImaginaryEnergyLevel(
                atom=self,
                uid='imaginary_level_3P0 J=1',
                config='combination',
                S=None, L=None, J=1,
                energy=43880.31))
        super().__init__(
            fine_energy_levels,
            nuclear_spin=self.__class__.nuclear_spin,
            nuclear_magneton=self.__class__.nuclear_magneton,
            B_field=B_field)

    def __str__(self):
        return self.__class__.__name__

    def load_fine_levels_from_json(self) -> List['FineEnergyLevel']:
        """Loads fine structure + metadata from JSON file."""
        filename = 'yb_level_structure.json'
        filepath = os.path.join(os.path.dirname(__file__), 'data', filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        fine_levels = []
        for item in data:
            if item['level_id'] == '6s6p 3P1':
                core_polarizability = get_3P1_core_polarizability
            else:
                try:
                    core_polarizability = item['core_polarizability'][0]['value']
                except KeyError:
                    core_polarizability = 0
            try:
                hyperfine_constant = item['magnetic_dipole_hyperfine_constant'][0]['value'] * 1e9
            except (KeyError, TypeError):
                hyperfine_constant = 0
            fine_level = FineEnergyLevel(
                atom=self,
                uid=item['level_id'],
                config=item['configuration'],
                energy=item['energy'][0]['value'],
                S=item['S'], L=item['L'], J=item['J'],
                lande_g=item.get('lande_g', None),
                core_polarizability=core_polarizability,
                hyperfine_constant=hyperfine_constant)
            fine_levels.append(fine_level)
        return fine_levels

    def load_reduced_matrix_elements(self):
        """Loads reduced matrix elements [atomic units] from JSON file."""
        filename = 'yb_matrix_elements.json'
        filepath = os.path.join(os.path.dirname(__file__), 'data', filename)
        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.load(file)
        matrix_elements = {}
        for item in data:
            level1 = item["level1"]
            level2 = item["level2"]
            if level1 in self.fine_level_ids and level2 in self.fine_level_ids and level1 != level2:
                # use the first value if multiple sources
                entry = item["rme"][0]
                # set symmetric values
                matrix_elements.setdefault(level1, {})[level2] = entry
                matrix_elements.setdefault(level2, {})[level1] = entry
        return matrix_elements
