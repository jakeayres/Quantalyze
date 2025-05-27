import numpy as np
from scipy.constants import hbar, pi
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
from scipy.constants import electron_mass, elementary_charge
from scipy.integrate import dblquad, quad, fixed_quad
import time

import matplotlib.pyplot as plt



class FermiSurface:

    
    def relaxation_time(self, phi):
        return 0.2e-12 - 0.02e-12 * np.cos(4 * phi)
    
    
    def effective_mass(self, phi):
        return 5.0 * electron_mass
    
    
    def cyclotron_frequency(self, phi, field):
        return elementary_charge * field / self.effective_mass(phi)
    
    
    def fermi_wavevector(self, phi):
        k00 = 0.728e10
        k40 = -3.3e-2 * k00
        return k00 + np.cos(4*phi) * k40
    
    
    def zeta(self, phi):
        dphi = 1e-6
        ln_kf_plus = np.log(self.fermi_wavevector(phi + dphi))
        ln_kf_minus = np.log(self.fermi_wavevector(phi - dphi))
        d_lnkf_dphi = (ln_kf_plus - ln_kf_minus) / (2 * dphi)
        return np.arctan(d_lnkf_dphi)
    
    
    def fermi_velocity(self, phi):
        return self.fermi_wavevector(phi) * hbar / self.effective_mass(phi)
    
    
    def fermi_velocity_x(self, phi):
        return self.fermi_wavevector(phi) * np.cos(phi - self.zeta(phi)) * hbar / self.effective_mass(phi)
    
    
    def fermi_velocity_y(self, phi):
        return self.fermi_wavevector(phi) * np.sin(phi - self.zeta(phi)) * hbar / self.effective_mass(phi)


    def damping_factor(self, theta, phi, field):
        integral = quad(
            lambda x: 1/(self.cyclotron_frequency(x, field) * self.relaxation_time(x)), 
            theta - phi, 
            theta
        )[0]
        return np.exp(-integral)
    
    
    def prefactor(self, field):
        numerator = elementary_charge**3 * field
        denominator = 2 * pi**2 * hbar**2 * 13e-10
        return numerator / denominator
    
    
    def integrand(self, theta, phi, field):
        numerator = self.fermi_velocity_x(theta) * self.fermi_velocity_x(theta - phi)
        denominator = self.cyclotron_frequency(theta, field) * self.cyclotron_frequency(theta - phi, field)
        damp = self.damping_factor(theta, phi, field)
        return damp * numerator / denominator
    
    
    def _find_damping_integration_limit(self, theta, field):
        phi_max = 1e-3
        while self.damping_factor(theta, phi_max, field) > 1e-4:
            phi_max *= 1.5
        return phi_max
    
    
    def _find_phi(self, theta, field, points=100):
        phi_max = self._find_damping_integration_limit(theta, field)
        phi_min = 1e-6
        phi = np.geomspace(phi_min, phi_max, points)
        phi = np.insert(phi, 0, 0)
        return phi

    
    def conductivity(self, field):
        theta = np.linspace(0, 2 * pi, 100)
        phi = self._find_phi(0, field, points=500)
        
        t0 = time.time()
        
        y = np.zeros((len(theta), len(phi)))
        for i, t in enumerate(theta):
            for j, p in enumerate(phi):
                y[i, j] = self.integrand(t, p, field)
                
        print(f"Time taken for integrand calculation: {time.time() - t0:.2f} seconds")
        
        return self.prefactor(field) * np.trapezoid(np.trapezoid(y, phi, axis=1), theta)
    
    
    def plot_relaxation_time(self, ax):
        phi = np.linspace(0, 2 * pi, 100)
        ax.plot(phi, self.relaxation_time(phi), label='Relaxation Time')
        ax.set_xlabel('Phi (radians)')
        ax.set_ylabel('Relaxation Time (s)')
        ax.legend()
        ax.grid()
        
        
    def plot_relaxation_rate(self, ax):
        phi = np.linspace(0, 2 * pi, 100)
        ax.plot(phi, 1 / self.relaxation_time(phi) * 1e-12, label='Relaxation Rate')
        ax.set_xlabel('Phi (radians)')
        ax.set_ylabel('Relaxation Rate (ps$^{-1}$)')
        ax.legend()
        ax.grid()
    




fs = FermiSurface()

fig, ax = plt.subplots(figsize=(10, 6))
fs.plot_relaxation_rate(ax)

fig, ax = plt.subplots(figsize=(10, 6))


for B in np.arange(1, 50, 10):
    
    phi = fs._find_phi(0, B)
    
    y = [fs.damping_factor(0, x, B) for x in phi]
    ax.plot(phi, y, '.-', label='Damping Factor')
    
    # y = [fs.damping_factor(np.pi/8, x, 1) for x in theta]
    # ax.plot(theta, y, label='Damping Factor')

plt.show()

print('Geomspace integration results:')
for B in np.arange(1, 50, 10):
    x = fs.conductivity(B)
    print(1/x * 1e8)