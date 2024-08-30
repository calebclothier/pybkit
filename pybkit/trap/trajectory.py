import numpy as np
import matplotlib.pyplot as plt
from functools import wraps


class Trajectory:
    
    def __init__(self, txyz: np.ndarray, trj_func, **trj_kwargs) -> None:
        self.txyz = txyz
        self.trj_func = trj_func
        self.trj_kwargs = trj_kwargs
        # get spacetime coordinates to traverse
        ts = txyz[:,0]
        xs = txyz[:,1]
        ys = txyz[:,2]
        zs = txyz[:,3]
        # piecewise trajectory position/velocity that passes through coordinates 
        x0_t = piecewise_func(ts, xs, self.trj_func, func_type='y', **self.trj_kwargs)
        y0_t = piecewise_func(ts, ys, self.trj_func, func_type='y', **self.trj_kwargs)
        z0_t = piecewise_func(ts, zs, self.trj_func, func_type='y', **self.trj_kwargs)
        self._trj_position = [x0_t, y0_t, z0_t]
        vx0_t = piecewise_func(ts, xs, self.trj_func, func_type='dydx', **self.trj_kwargs)
        vy0_t = piecewise_func(ts, ys, self.trj_func, func_type='dydx', **self.trj_kwargs)
        vz0_t = piecewise_func(ts, zs, self.trj_func, func_type='dydx', **self.trj_kwargs)
        self._trj_velocity = [vx0_t, vy0_t, vz0_t]
        # compute gradients for acceleration and jerk
        # self._trj_acceleration = np.gradient(self.get_position(ts), ts)
        
    def x(self, t):
        return self._trj_position[0](t)
    
    def y(self, t):
        return self._trj_position[1](t)
    
    def z(self, t):
        return self._trj_position[2](t)
    
    def vx(self, t):
        return self._trj_velocity[0](t)
    
    def vy(self, t):
        return self._trj_velocity[1](t)
    
    def vz(self, t):
        return self._trj_velocity[2](t)
        
    def get_position(self, t: float):
        return np.array([self.x(t), self.y(t), self.z(t)]).T
    
    def get_velocity(self, t: float):
        return np.array([self.vx(t), self.vy(t), self.vz(t)]).T
    
    def plot_position(self):
        t0, tf = np.min(self.txyz[:,0]), np.max(self.txyz[:,0])
        ts = np.linspace(t0, tf, 1000)
        positions = self.get_position(ts)
        fig, ax = plt.subplots(ncols=3, figsize=(10,4), sharey=True)
        ax[0].plot(1e6*ts, 1e6*positions[:,0])
        ax[0].set_title('X position')
        ax[0].set_ylabel('Position [um]')
        ax[0].set_xlabel('Time [us]')
        ax[1].plot(1e6*ts, 1e6*positions[:,1])
        ax[1].set_title('Y position')
        ax[1].set_xlabel('Time [us]')
        ax[2].plot(1e6*ts, 1e6*positions[:,2])
        ax[2].set_title('Z position')
        ax[2].set_xlabel('Time [us]')
        fig.suptitle('Trap trajectory: position', weight='bold')
        fig.tight_layout()
        
    def plot_velocity(self):
        t0, tf = np.min(self.txyz[:,0]), np.max(self.txyz[:,0])
        ts = np.linspace(t0, tf, 1000)
        velocities = self.get_velocity(ts)
        fig, ax = plt.subplots(ncols=3, figsize=(10,4), sharey=True)
        ax[0].plot(1e6*ts, velocities[:,0])
        ax[0].set_title('X velocity')
        ax[0].set_ylabel('Velocity [um/us]')
        ax[0].set_xlabel('Time [us]')
        ax[1].plot(1e6*ts, velocities[:,1])
        ax[1].set_title('Y velocity')
        ax[1].set_xlabel('Time [us]')
        ax[2].plot(1e6*ts, velocities[:,2])
        ax[2].set_title('Z velocity')
        ax[2].set_xlabel('Time [us]')
        fig.suptitle('Trap trajectory: velocity', weight='bold')
        fig.tight_layout()
        

def polynomial_connect(coeff_func):
    @wraps(coeff_func)
    def trj_func(x, x0, x1, y0, y1, **kwargs):
        a_list = coeff_func(**kwargs)
        y = y0 + sum([a * (y1 - y0) * ((x - x0) / (x1 - x0)) ** n for n, a in enumerate(a_list, start=0)])
        dydx = sum([a * n * (y1 - y0) * (x - x0) ** (n - 1) / (x1 - x0) ** n for n, a in enumerate(a_list[1:], start=1)])
        return {'y': y, 'dydx': dydx}
    return trj_func

@polynomial_connect
def min_jerk():
    return [0, 0, 0, 10, -15, 6]

@polynomial_connect
def const_jerk():
    return [0, 0, 3, -2]

@polynomial_connect
def drag_out():
    return [0, 0, 2.5, 0, -2.5, 1]

@polynomial_connect
def arb_fifth_poly(beta):
    return [0, 0, 15 - 8 * beta, -50 + 32 * beta, 60 - 40 * beta, -24 + 16 * beta]

@polynomial_connect
def zero_jerk():
    return [0, 0, 0, 0, 35, -84, 70, -20]

@polynomial_connect
def arb_eighth_poly(beta):
    return [0, 0, 0, 70 - 32 * beta, -315 + 160 * beta, 546 - 288 * beta, -420 + 224 * beta, 120 -64*beta]

@polynomial_connect
def arb_seven_poly_2d(alpha, beta):
    return [0, 0, 35-24*alpha-2*beta, 
            2*(-105+80*alpha+8*beta), 
            -10*(-56+44*alpha+5*beta),
            -784+624*alpha+76*beta, 
            -56*(-10+8*alpha+beta),
            16*(-10+8*alpha+beta)]

@polynomial_connect
def linear():
    return [0, 1]

def piecewise_func(x_list, y_list, f_connect, func_type='y', **connect_kwargs):
    """ Connect discrete points using f_connect """
    if func_type == 'y':
        y_min = y_list[0]
        y_max = y_list[-1]
    elif func_type == 'dydx':
        y_min = 0
        y_max = 0
    else:
        raise TypeError('*****')

    if len(x_list) == 1:
        f_x = lambda x: y_min
    elif len(x_list) >= 2:
        def f_x(x):
            if x < x_list[0]:
                return y_min
            elif x >= x_list[-1]:
                return y_max
            else:
                for i in range(len(x_list) - 1):
                    if x_list[i] <= x < x_list[i + 1]:
                        return f_connect(x, x_list[i], x_list[i + 1], y_list[i], y_list[i + 1], **connect_kwargs)[func_type]
    else:
        raise RuntimeError('*****')
    return lambda x: np.squeeze(list(map(f_x, np.ravel(x))))