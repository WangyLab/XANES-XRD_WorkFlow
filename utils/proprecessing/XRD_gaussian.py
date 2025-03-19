from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import wofz

cif_path = "Material.cif"
structure = Structure.from_file(cif_path)
xrd_calculator = XRDCalculator()
xrd_pattern = xrd_calculator.get_pattern(structure)

two_theta = xrd_pattern.x
intensities = xrd_pattern.y

def voigt(x, amp, pos, sigma, gamma):
    z = ((x - pos) + 1j * gamma) / (sigma * np.sqrt(2))
    return amp * np.real(wofz(z)) / (sigma * np.sqrt(2 * np.pi))

def dynamic_sigma(theta, a=0.04, b=0.05):
    fwhm = a * np.sqrt(theta) + b
    sigma = fwhm / 2.355
    gamma = fwhm / 3
    return sigma, gamma

x_dense = np.linspace(10, 90, 801)
y_smooth = np.zeros_like(x_dense)

for pos, amp in zip(two_theta, intensities):
    if 10 <= pos <= 90:
        sigma, gamma = dynamic_sigma(pos)
        y_smooth += voigt(x_dense, amp * (1 + 0.5 * (90 - pos) / 90), pos, sigma, gamma)

gaussian_noise_std = 10
poisson_noise_factor = 0.005

y_noisy = y_smooth + np.random.normal(0, gaussian_noise_std, y_smooth.shape)

plt.figure(figsize=(8, 6))
plt.stem(two_theta, y_noisy, linefmt="b-", markerfmt=" ", basefmt=" ")
plt.xlabel("2θ (degrees)")
plt.ylabel("Intensity")
plt.title("Simulated XRD Pattern")
plt.grid(False)
plt.savefig("xrd_gau.png", dpi=600, bbox_inches='tight')
plt.show()