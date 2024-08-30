import numpy as np
import matplotlib.pyplot as plt

from ..amo.laser import GaussianLaser
from .geometry import Vector3D
from .tweezers import Tweezer, TweezerGroup


class SLMDevice:

    def __init__(self, laser: GaussianLaser):
        self.laser = laser
        self.tweezer_group = None

    def generate_tweezers(self, positions: list[Vector3D]) -> TweezerGroup:
        tweezers = [Tweezer(position, self.laser) for position in positions]
        self.tweezer_group = TweezerGroup(tweezers, generator=self)
        return self.tweezer_group

    def __repr__(self):
        return f"SLMDevice: laser={self.laser}"


class AODDevice:

    def __init__(self,laser: GaussianLaser):
        self.laser = laser
        self.tweezer_group = None
        self.row_freqs = None
        self.col_freqs = None
        self.spacing = None

    def generate_tweezers(self, row_freqs: list, col_freqs: list, spacing: float) -> TweezerGroup:
        tweezers = [Tweezer(Vector3D(fx, fy, 0), self.laser) for fx in col_freqs for fy in row_freqs]
        self.tweezer_group = TweezerGroup(tweezers, generator=self)
        self.row_freqs = row_freqs
        self.col_freqs = col_freqs
        self.spacing = spacing
        return self.tweezer_group

    def plot_tweezers_phonon_dispersion(self):
        thetas = np.linspace(-np.pi/100, np.pi/100, 1000)
        thetas = np.append(thetas, np.linspace(np.pi/2-np.pi/100, np.pi/2+np.pi/100, 1000))
        velocity = self.acoustic_velocity(thetas)
        slowness = 1/velocity
        xs = slowness * np.cos(thetas)
        ys = slowness * np.sin(thetas)
        max_val = np.max([np.max(xs), np.max(ys)])
        plt.figure(figsize=(7,7))
        min_x, max_x = np.inf, -np.inf
        min_y, max_y = np.inf, -np.inf
        for tweezer in self.tweezer_group.tweezers:
            pos = tweezer.position
            if pos.x < min_x:
                min_x = pos.x
            if pos.x > max_x:
                max_x = pos.x
            if pos.y < min_y:
                min_y = pos.y
            if pos.y > max_y:
                max_y = pos.y
            plt.scatter(xs*pos.x/max_val, ys*pos.x/max_val + pos.y, c='C0', alpha=0.1, s=2)
            plt.scatter(xs*pos.y/max_val + pos.x, ys*pos.y/max_val, c='C4', alpha=0.1, s=2)
            plt.scatter(pos.x, pos.y, c='C1', s=20)
        xscale = max_x - min_x
        yscale = max_y - min_y
        plt.xlim(min_x - 0.8*xscale, max_x + 0.8*xscale)
        plt.ylim(min_y - 0.8*yscale, max_y + 0.8*yscale)

    def plot_tweezers_intermodulation(self, dac_amp=1, offset_dB=28, decay_distance=10, weighted=True, min_dB=None):

        row_freqs, row_powers = estimate_intermod_powers(
            freqs=self.row_freqs,
            dac_amp=dac_amp,
            offset_dB=offset_dB,
            decay_distance=decay_distance,
            weighted=weighted)
        col_freqs, col_powers = estimate_intermod_powers(
            freqs=self.col_freqs,
            dac_amp=dac_amp,
            offset_dB=offset_dB,
            decay_distance=decay_distance,
            weighted=weighted)

        if len(col_freqs) == 0:
            xmin = self.col_freqs[0]
            xmax = xmin + self.spacing
            xs = [xmin]
        else:
            xmin, xmax = np.min(col_freqs), np.max(col_freqs)
            xs = np.arange(xmin, xmax + self.spacing + 1e-6, self.spacing)

        if len(row_freqs) == 0:
            ymin = self.row_freqs[0]
            ymax = ymin + self.spacing
            ys = [ymin]
        else:
            ymin, ymax = np.min(row_freqs), np.max(row_freqs)
            ys = np.arange(ymin, ymax + self.spacing + 1e-6, self.spacing)

        N, M = len(ys), len(xs)
        result = np.zeros(shape=(N, M))

        rounded_row_freqs = [round(f,3) for f in self.row_freqs]
        rounded_col_freqs = [round(f,3) for f in self.col_freqs]

        for i, fx in enumerate(xs):
            if round(fx, 3) in col_freqs:
                idx = np.argwhere(col_freqs == round(fx, 3))[0][0]
                power = col_powers[idx]
                if min_dB:
                    power = power if power > min_dB else min_dB
                for j, fy in enumerate(ys):
                    if round(fy, 3) not in rounded_row_freqs and round(fx, 3) not in rounded_col_freqs:
                        try:
                            approx_power = row_powers[np.argwhere(row_freqs == round(fy, 3))[0][0]]
                            approx_power += col_powers[np.argwhere(col_freqs == round(fx, 3))[0][0]]
                            approx_power = approx_power if approx_power > min_dB else min_dB
                            result[N-i-1,j] = approx_power #min_dB
                        except:
                            result[N-i-1,j] = None
                    if round(fy, 3) in rounded_row_freqs and round(fx, 3) not in rounded_col_freqs:
                        result[N-j-1,i] = power
        for i, fy in enumerate(ys):
            if round(fy, 3) in row_freqs:
                idx = np.argwhere(row_freqs == round(fy, 3))[0][0]
                power = row_powers[idx]
                power = power if power > min_dB else min_dB
                for j, fx in enumerate(xs):
                    # if round(fy, 3) not in rounded_row_freqs and round(fx, 3) not in rounded_col_freqs:
                    #     try:
                    #         approx_power = row_powers[np.argwhere(row_freqs == round(fy, 3))[0][0]]
                    #         approx_power += col_powers[np.argwhere(col_freqs == round(fx, 3))[0][0]]
                    #         approx_power = approx_power if approx_power > min_dB else min_dB
                    #         result[N-i-1,j] = approx_power #min_dB
                    #     except:
                    #         result[N-i-1,j] = None
                    if round(fx, 3) in rounded_col_freqs and round(fy, 3) not in rounded_row_freqs:
                        result[N-i-1,j] = power


        result = np.ma.masked_values(result, value=0)


        plt.figure(figsize=(7,7))
        pad = self.spacing / 2.
        im = plt.imshow(result, extent=[xmin-pad,xmax+pad+self.spacing,ymin-pad,ymax+pad+self.spacing])
        plt.xlabel('Column frequency [MHz]')
        plt.ylabel('Row frequency [MHz]')
        cbar = plt.colorbar(im, label='Relative power [dB]')
        ax = plt.gca()
        ax_pos = ax.get_position()
        cbar.ax.set_position([ax_pos.x1 + 0.01, ax_pos.y0, 0.03, ax_pos.height])

        for tweezer in self.tweezer_group.tweezers:
            pos = tweezer.position
            plt.scatter(pos.x, pos.y, c='C1')

        plt.xticks(np.arange(xmin-pad, xmax+pad+self.spacing+1e-3, 2*self.spacing), rotation=-45)
        plt.yticks(np.arange(ymin-pad, ymax+pad+self.spacing+1e-3, 2*self.spacing))
        plt.xticks(np.arange(xmin-pad, xmax+pad+self.spacing+1e-3, self.spacing), rotation=-45, minor=True)
        plt.yticks(np.arange(ymin-pad, ymax+pad+self.spacing+1e-3, self.spacing), minor=True)

        plt.grid(which='both')
        plt.xlim(xmin-pad, xmax+pad)
        plt.ylim(ymin-pad, ymax+pad)
        plt.title('Predicted intermod rejection [dBc]')



    def acoustic_velocity(self, theta):
        c11 = 5.59e10
        c12 = 5.13e10
        c66 = 6.62e10
        rho = 6.02e3
        v_acoustic = np.sqrt((1/(2*rho)) * ((c11 + c66) - np.sqrt((c66-c11)**2 * np.cos(2*theta + np.pi/2)**2 + (c12+c66)**2 * np.sin(2*theta + np.pi/2)**2)))
        return v_acoustic

    def __repr__(self):
        return f"AODDevice: laser={self.laser}"




def gaussian(x, mu, sigma):
    res = np.exp(-((x - mu)**2) / (2 * sigma**2))
    return res / res.sum()

def exp_weight(formula, lam=10):
    d01 = abs(abs(formula[0]) - abs(formula[1]))
    d02 = abs(abs(formula[0]) - abs(formula[2]))
    d12 = abs(abs(formula[1]) - abs(formula[2]))
    weight = np.exp(-(d01+d02+d12)/lam)
    return weight

def calculate_third_order_intermods(freqs, weighting=True, decay_distance=10):
    N = len(freqs)
    intermod_products = {}
    weighted_products = {}
    for i in range(N):
        for j in range(N):
            for k in range(N):
                products = []
                if i != k and j != k and i != j:
                    products.append([round(freqs[i] + freqs[j] - freqs[k], 3), (freqs[i], freqs[j], -freqs[k])])
                    products.append([round(freqs[i] - freqs[j] + freqs[k], 3), (freqs[i], -freqs[j], freqs[k])])
                    products.append([round(-freqs[i] + freqs[j] + freqs[k], 3), (-freqs[i], freqs[j], freqs[k])])
                elif i != j:
                    products.append([round(2*freqs[i] - freqs[j], 3), (freqs[i], freqs[i], -freqs[j])])
                    # products.append([2*freqs[j] - freqs[i], (freqs[j], freqs[j], -freqs[i])])
                for product, formula in products:
                    sorted_formula = tuple(sorted(list(formula), key=abs))
                    if product in intermod_products:
                        intermod_products[product].add(sorted_formula)
                        weighted_products[product] += exp_weight(formula, lam=decay_distance)
                    else:
                        intermod_products[product] = {sorted_formula}
                        weighted_products[product] = exp_weight(formula, lam=decay_distance)
    return intermod_products, weighted_products


def estimate_intermod_powers(freqs, dac_amp=1, offset_dB=28, decay_distance=10, weighted=True):
    unweighted_products, weighted_products = calculate_third_order_intermods(freqs, decay_distance=decay_distance)
    num_freqs = len(freqs)
    power_scale = 1 / np.sqrt(num_freqs) # RFSoC amplitude scaling
    dac_scale = dac_amp**4 # intermod power scales as dac_amp^6, main power scales as dac_amp^2
    intermod_freqs = list(sorted(unweighted_products.keys()))
    unweighted_counts = dac_scale * power_scale**2 * np.array([len(set(unweighted_products[f])) for f in intermod_freqs])
    weighted_counts = dac_scale * power_scale**2 * np.array([weighted_products[f] for f in intermod_freqs])
    mask = ~np.in1d(intermod_freqs, freqs)
    indices = np.where(mask)[0]
    masked_uw_counts = unweighted_counts[indices]
    masked_w_counts = weighted_counts[indices]
    uw_power_dBc = 10*np.log10(masked_uw_counts)-offset_dB
    w_power_dBc = 10*np.log10(masked_w_counts)-offset_dB
    if weighted:
        return np.array(intermod_freqs)[indices], w_power_dBc
    else:
        return np.array(intermod_freqs)[indices], uw_power_dBc

