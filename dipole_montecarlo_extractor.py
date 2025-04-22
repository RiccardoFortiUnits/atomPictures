
from scipy.interpolate import interp1d
import numpy as np

def montecarlo_1D(pdf, ranges, steps):
	#use this distribution generator for 1D probability density functions (PDF)
	#the generated extractor will return a random value x that follows the given PDF

	grid = np.linspace(ranges[0], ranges[1], 1+int(np.ceil((ranges[1]-ranges[0])/steps)))

	pdf_values = pdf(grid)
	cdf_values_x = np.cumsum(pdf_values)
	cdf_values_x -= cdf_values_x[0]
	cdf_values_x /= cdf_values_x[-1]
	interp = interp1d(cdf_values_x, grid, kind='linear', fill_value="extrapolate")
	def get_x():
		rand = np.random.random()
		x = interp(rand)
		return x
	return get_x

def qPolarizationAnglePDF(theta):
	return 3+np.cos(2*theta)

qPolarizationExtractor = montecarlo_1D(qPolarizationAnglePDF, (-np.pi/2, np.pi/2), np.pi/200)
def SpontaneousEmission_qPolarization (wavelength,m):
    """
    Emit a photon of given wavelength in a direction that follows the dipole scattering distribution
    """
    kmod = 2*np.pi/(wavelength)
    k = np.zeros(3)
    theta = qPolarizationExtractor()
    phi = np.random.uniform(0,2*np.pi)
    k[0] = kmod*np.cos(phi)*np.cos(theta)
    k[2] = kmod*np.sin(phi)*np.cos(theta)
    k[1] = kmod*np.sin(theta)
    return hbar*k/m