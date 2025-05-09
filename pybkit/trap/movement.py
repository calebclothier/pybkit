import numpy as np
import matplotlib.pyplot as plt
from scipy import constants
from scipy.integrate import odeint
import multiprocess as mp
from tqdm.notebook import tqdm

from pybkit.amo.atom import AtomSpecies, HyperfineEnergyLevel
from pybkit.trap.tweezers import Tweezer
from pybkit.trap.trajectory import Trajectory


class TweezerMove:

    def __init__(
        self, 
        tweezer: Tweezer, 
        atom: AtomSpecies,
        level: HyperfineEnergyLevel,
        txyz: np.ndarray,
        dxdt2z: float, 
        trj_func, 
        **trj_kwargs
    ):
        self.tweezer = tweezer
        self.atom = atom
        self.level = level
        self.txyz = txyz
        self.dxdt2z = dxdt2z
        self.traj = Trajectory(txyz, trj_func, **trj_kwargs)
        
    def get_trajectory_potential(self, t, x, y, z):
        return self.tweezer.trap_potential(
            self.atom, 
            self.level, 
            x - self.traj.x(t), 
            y - self.traj.y(t),
            z - self.traj.z(t), 
            self.dxdt2z * self.traj.vx(t), 
            self.dxdt2z * self.traj.vy(t))
        
    def get_trajectory_force(self, t, x, y, z):
        return self.tweezer.trap_force(
            self.atom, 
            self.level, 
            x - self.traj.x(t), 
            y - self.traj.y(t),
            z - self.traj.z(t), 
            self.dxdt2z * self.traj.vx(t), 
            self.dxdt2z * self.traj.vy(t))
        
    def sample_maxwell_boltzmann(
        self,
        num_samples: float, 
        T: float,
        avg_position: np.ndarray
    ):
        radial_freq = self.tweezer.radial_trap_frequency(self.atom, self.level)
        axial_freq = self.tweezer.axial_trap_frequency(self.atom, self.level)
        sigma_v = np.sqrt(constants.k * T / self.atom.mass)
        sigma_r = np.sqrt(constants.k * T / self.atom.mass) / radial_freq
        sigma_z = np.sqrt(constants.k * T / self.atom.mass) / axial_freq
        position = np.array([
            np.random.normal(loc=avg_position[0], scale=sigma_r, size=num_samples),
            np.random.normal(loc=avg_position[1], scale=sigma_r, size=num_samples),
            np.random.normal(loc=avg_position[2], scale=sigma_z, size=num_samples)])
        velocity = np.random.normal(loc=0, scale=sigma_v, size=(3, num_samples))
        return position.T, velocity.T
    
    def calculate_energy(
        self,
        t: float,
        position: np.ndarray,
        velocity: np.ndarray
    ):
        kinetic_energy = 0.5 * self.atom.mass * np.linalg.norm(velocity, axis=1)**2
        potential_energy = self.get_trajectory_potential(t, position[:,0], position[:,1], position[:,2])
        energy = kinetic_energy + potential_energy
        energy = np.array(energy, dtype=float)
        return energy
        
    def simulate_atom_dynamics(
        self, 
        t: np.ndarray, 
        p0: np.ndarray,
        v0: np.ndarray,
        plot=True
    ):
        def derivative(t, var):
            x, y, z, vx, vy, vz = var
            ax, ay, az = self.get_trajectory_force(t, x, y, z) / self.atom.mass
            dvar_dt = [vx, vy, vz, ax, ay, az]
            return dvar_dt
        sol = odeint(func=derivative, y0=np.array([*p0, *v0]), t=t, tfirst=True)
        position, velocity = sol[:,:3], sol[:,3:]
        if plot:
            self.plot_atom_dynamics(t, position, velocity)
        return position, velocity
    
    def simulate_atom_dynamics_parallel(
        self, 
        t: np.ndarray, 
        p0s: np.ndarray, 
        v0s: np.ndarray
    ):
        num_samples = np.array(p0s).shape[0]
        def _simulate(ample_idx):
            return self.simulate_atom_dynamics(t, p0s[ample_idx,:], v0s[ample_idx,:], plot=False)
        num_workers = mp.cpu_count()  
        with mp.Pool(num_workers) as p:
            sol_list = list(tqdm(p.imap(_simulate, range(num_samples)), total=num_samples))
        return sol_list
    
    def simulate_ensemble_dynamics(
        self, 
        t: np.ndarray, 
        T: float, 
        num_samples: int, 
        plot: bool = True, 
        num_bins: int = 30
    ):
        # sample initial positions and velocities
        p_initial, v_initial = self.sample_maxwell_boltzmann(
            num_samples=num_samples, T=T, avg_position=self.txyz[0,:])
        energy_initial = self.calculate_energy(t[0], p_initial, v_initial)
        energy_initial /= (constants.h * 1e6) # convert to MHz
        # simulate dynamics
        sol_list = self.simulate_atom_dynamics_parallel(t, p_initial, v_initial)
        p_final = np.array([s[0][-1,:] for s in sol_list])
        v_final = np.array([s[1][-1,:] for s in sol_list])
        energy_final = self.calculate_energy(t[-1], p_final, v_final)
        energy_final /= (constants.h * 1e6)
        idxs = np.argwhere(energy_final >= 0)
        trap_depth = self.tweezer.trap_depth(self.atom, self.level, unit='MHz') / (2*np.pi)
        trap_position = [self.traj.x(t), self.traj.y(t), self.traj.z(t)]
        trap_velocity = [self.traj.vx(t), self.traj.vy(t), self.traj.vz(t)]
        # plot trajectories
        if plot:
            alpha = 1 / np.sqrt(num_samples)
            fig, ax = plt.subplots(nrows=6, ncols=3, sharex=True, figsize=(9,8))
            for i, s in enumerate(sol_list):
                position, velocity = s
                labels = ['X', 'Y', 'Z']
                for j, label in enumerate(labels):
                    ax[0,j].plot(t*1e6, trap_position[j]*1e6, color=f'C{j}')
                    ax[0,j].set_title(f'Trap {label} position')
                    ax[1,j].plot(t*1e6, trap_velocity[j], color=f'C{j}')
                    ax[1,j].set_title(f'Trap {label} velocity')
                    ax[2,j].plot(t*1e6, position[:,j]*1e6, color=f'C{j}', alpha=alpha)
                    # ax[2,j].scatter(t[-1]*1e6, position[-1,j]*1e6, color=f'C{j}')
                    ax[2,j].set_title(f'Atom {label} position')
                    ax[3,j].plot(t*1e6, velocity[:,j], color=f'C{j}', alpha=alpha)
                    # ax[3,j].scatter(t[-1]*1e6, velocity[-1,j], color=f'C{j}')
                    ax[3,j].set_title(f'Atom {label} velocity')
                    ax[4,j].plot(t*1e6, (position[:,j]-trap_position[j])*1e9, color=f'C{j}', alpha=alpha)
                    # ax[4,j].scatter(t[-1]*1e6, (position[-1,j]-trap_position[j][-1])*1e9, color=f'C{j}')
                    ax[4,j].set_title(f'Atom - Trap {label} position')
                    ax[5,j].plot(t*1e6, (velocity[:,j]-trap_velocity[j])*1e3, color=f'C{j}', alpha=alpha)
                    # ax[5,j].scatter(t[-1]*1e6, (velocity[-1,j]-trap_velocity[j][-1])*1e3, color=f'C{j}')
                    ax[5,j].set_title(f'Atom - Trap {label} velocity')
                ax[0,0].set_ylabel('um')
                ax[1,0].set_ylabel('um/us')
                ax[2,0].set_ylabel('um')
                ax[3,0].set_ylabel('um/us')
                ax[4,0].set_ylabel('nm')
                ax[5,0].set_ylabel('nm/us')
                ax[5,0].set_xlabel('Time [us]')
                ax[5,1].set_xlabel('Time [us]')
            fig.legend(loc='upper center', ncols=3, bbox_to_anchor=(0., 0.975, 1.0, 0.))
            fig.suptitle('Trap & simulated atom trajectories', weight='bold', y=1.)
            fig.tight_layout()
            # plot initial/final energy histograms
            plt.figure()
            Emin, Emax = min(min(energy_initial), min(energy_final)), max(max(energy_initial), max(energy_final))
            if Emax > 0:
                bins = np.concatenate([np.linspace(Emin, 0, num_bins), np.linspace(0, Emax, int(num_bins * abs(Emax / Emin)))])
            else:
                bins = np.linspace(Emin, 0, num_bins)
            plt.hist(energy_initial, bins=bins, color='C0', alpha=0.5, label='initial')
            plt.hist(energy_final, bins=bins, color='C1', alpha=0.5, label='final')
            ylim = plt.ylim()
            plt.vlines(-trap_depth, *ylim, linestyle='--', color='C2', label='trap depth', zorder=999)
            plt.vlines(0, *ylim, linestyle='--', color='black', label='atom loss threshold', zorder=999)
            plt.xlabel('Energy [MHz]')
            plt.ylabel('Counts')
            plt.yscale('log')
            plt.legend()
        return energy_initial, energy_final
    
    def plot_atom_dynamics(
        self, 
        t: np.ndarray, 
        position: np.ndarray, 
        velocity: np.ndarray
    ):
        fig, ax = plt.subplots(nrows=3, ncols=2, sharex=True, figsize=(8,7))
        ax[0,0].plot(t*1e6, self.traj.x(t)*1e6, label='x')
        ax[0,0].plot(t*1e6, self.traj.y(t)*1e6, label='y')
        ax[0,0].plot(t*1e6, self.traj.z(t)*1e6, label='z')
        ax[0,0].set_title('Trap position')
        ax[0,0].set_ylabel('Position [um]')
        ax[0,1].plot(t*1e6, self.traj.vx(t))
        ax[0,1].plot(t*1e6, self.traj.vy(t))
        ax[0,1].plot(t*1e6, self.traj.vz(t))
        ax[0,1].set_title('Trap velocity')
        ax[0,1].set_ylabel('Velocity [um/us]')
        ax[1,0].plot(t*1e6, position[:,0]*1e6)
        ax[1,0].plot(t*1e6, position[:,1]*1e6)
        ax[1,0].plot(t*1e6, position[:,2]*1e6)
        ax[1,0].set_title('Atom position')
        ax[1,0].set_ylabel('Position [um]')
        ax[1,1].plot(t*1e6, velocity[:,0])
        ax[1,1].plot(t*1e6, velocity[:,1])
        ax[1,1].plot(t*1e6, velocity[:,2])
        ax[1,1].set_title('Atom velocity')
        ax[1,1].set_ylabel('Velocity [um/us]')
        ax[2,0].plot(t*1e6, (position[:,0]-self.traj.x(t))*1e9)
        ax[2,0].plot(t*1e6, (position[:,1]-self.traj.y(t))*1e9)
        ax[2,0].plot(t*1e6, (position[:,2]-self.traj.z(t))*1e9)
        ax[2,0].set_title('Atom relative position')
        ax[2,0].set_ylabel('Position [nm]')
        ax[2,1].plot(t*1e6, (velocity[:,0]-self.traj.vx(t))*1e3)
        ax[2,1].plot(t*1e6, (velocity[:,1]-self.traj.vy(t))*1e3)
        ax[2,1].plot(t*1e6, (velocity[:,2]-self.traj.vz(t))*1e3)
        ax[2,1].set_title('Atom relative velocity')
        ax[2,1].set_ylabel('Velocity [nm/us]')
        ax[2,0].set_xlabel('Time [us]')
        ax[2,1].set_xlabel('Time [us]')
        fig.legend(loc='upper center', ncols=3, bbox_to_anchor=(0., 0.975, 1.0, 0.))
        fig.suptitle('Trap & simulated atom trajectories', weight='bold', y=1.)
        fig.tight_layout()
        
    def plot_potential_along_trajectory(self):
        depth = self.tweezer.trap_depth(self.atom, self.level, unit='J')
        dr = np.sqrt(depth / (0.5 * self.atom.mass * self.tweezer.radial_trap_frequency(self.atom, self.level)**2))
        dz = np.sqrt(depth / (0.5 * self.atom.mass * self.tweezer.axial_trap_frequency(self.atom, self.level)**2))
        ts = np.linspace(0, 0.5, 20) * (self.txyz[-1,0] - self.txyz[0,0])
        rs = np.linspace(-2*dr/1e-6, 2*dr/1e-6, 1000) * 1e-6
        zs = np.linspace(-3*dz/1e-6, 3*dz/1e-6, 1000) * 1e-6
        fig, ax = plt.subplots(ncols=3, figsize=(10,4), sharey=True)
        for i, t in enumerate(ts):
            x = self.traj.x(t)
            y = self.traj.y(t)
            z = self.traj.z(t)
            Ux = self.get_trajectory_potential(t, x + rs, y, z) / constants.hbar
            Uy = self.get_trajectory_potential(t, x, y + rs, z) / constants.hbar
            Uz = self.get_trajectory_potential(t, x, y, z + zs) / constants.hbar
            color = plt.cm.coolwarm(i / len(ts))
            ax[0].plot(rs * 1e6, Ux / 1e6, color=color)
            ax[1].plot(rs * 1e6, Uy / 1e6, color=color)
            ax[2].plot(zs * 1e6, Uz / 1e6, color=color)
        ax[0].set_ylabel('Trap potential [MHz]')
        ax[0].set_xlabel('Displacement [um]')
        ax[1].set_xlabel('Displacement [um]')
        ax[2].set_xlabel('Displacement [um]')
        ax[0].set_title('X direction')
        ax[1].set_title('Y direction')
        ax[2].set_title('Z direction')
        fig.suptitle('Trap potential astigmatism: trajectory from $t = 0$ to $t = \\tau_{move} / 2$', weight='bold')
        fig.tight_layout()