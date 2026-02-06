## ArQuS Lab Analysis library
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.optimize import curve_fit

def install_and_import(package):
	import subprocess
	import sys
	try:
		__import__(package)
	except ImportError:
		try:
			subprocess.check_call([sys.executable, "-m", "pip", "install", package])
		except:
			subprocess.check_call([sys.executable, "-m", "pip", "install", f"py{package}"])
		__import__(package)
		
install_and_import("astropy")
from astropy.stats import binom_conf_interval
from scipy import constants, signal
import scipy.optimize as opt
from scipy.integrate import quad
from scipy.stats import poisson as Poisson
from scipy.stats import mode
import h5py
import os
from numba import njit
from joblib import Parallel, delayed

############## Global parameters #################
ROI_ROWS = [4, 105] #[6,109]
ROI_COLS = [9, 19]# [12,21]
ORCA_ROI_ROWS = [15, 165]
ORCA_ROI_COLS = [15, 30]
SUB_ELECTRON_GAIN = 0.016
SENSITIVITY_GAIN = 0.276
N_TWEEZERS = 10
THRESHOLD = 10
##################################################
##################################################

######################################################################
##########################  Functions  ###############################
######################################################################

# @njit(nopython=True)
def Gaussian (x: np.ndarray, A: float, x0: float, sigma: float, offset: float):
	"""
	1D Gaussian function, not normalized: `A exp{-(x-x_0)^2/(2*sigma^2)} + offset`

	:params x: independent variable
	:type x: np.ndarray[float]
	:params A: amplitude of gaussian function
	:type A: float
	:params x0: center of gaussian function
	:type x0: float
	:params sigma: width of gaussian function
	:type sigma: float
	:params offset: baseline of gaussian function
	:type offset: float
	:return: return an array with the gaussian evaluated in x
	:rtype: np.ndarray[float]
	"""
	return A*np.exp(-(x-x0)**2/(2*sigma**2)) + offset

# @njit(nopython=True)
def double_Gaussian (x, x01, sigma1, x02, sigma2, A1, A2):
	"""
	Double gaussian with no offset.
	x: array
	x01, x02: centers. dtype = float
	sigma1, sigma2: standard deviations. dtype = float
	A1, A2: amplitudes. dtype = float
	"""
	return Gaussian(x, A1, x01, sigma1, 0) + Gaussian(x, A2, x02, sigma2, 0)

# @njit(nopython=True)
def normalized_Gaussian (x, x0, sigma):
	"""
	Normalized 1D Gaussian.
	x: array
	A: amplitude. dtype = float
	x_0: center. dtype = float
	sigma: standard deviation. dtype = float
	"""
	return Gaussian(x, 1, x0, sigma, 0) /(sigma*np.sqrt(2*np.pi))

# @njit(nopython=True)
def normalized_double_Gaussian (x, x01, sigma1, x02, sigma2, frac_prob):
	"""
	Normalized sum of 2 1D Gaussians.
	x: array
	x01, x02: centers. dtype = float
	sigma1, sigma2: standard deviation. dtype = float
	fractional_probablity: amplitude of first gaussian (second has 1-fractional_probability, to be normalized), dtype = float
	"""
	return ( frac_prob * normalized_Gaussian(x, x01, sigma1) + (1 - frac_prob) * normalized_Gaussian(x, x02, sigma2) )

# @njit(nopython=True)
def Gaussian_2D (xdata_tuple, A, x0, y0, sigma_x, sigma_y, offset):
	"""
	2D Gaussian function, not normalized: `A*exp{(x-x0)^2/(2*sigma_x^2) - (y-y_0)^2/(2*sigma_y^2)} + offset`\n
	:params x_data_tuple: independent variables, dtype = tuple of numpy array
	:params A: gaussian amplitude, dtype = float
	:params x0, y0: gaussian centers, dtype = float
	:params sigma: gaussian standard deviation, dtype = float
	:params offset: gaussian baseline, dtype = float
	"""
	(x, y) = xdata_tuple
	G = A*Gaussian(x,1,x0,sigma_x,0)*Gaussian(y,1,y0,sigma_y,0) + offset
	return G.ravel()

def double_Poisson(x, mu1, mu2, A):
	"""
	Sum of 2 poissonian distributions: `A1*exp{-mu1}*mu1^{x}/x! + A2*exp{-mu2}*mu2^{x}/x!`\n
	:params x_data_tuple: independent variables, dtype = tuple of numpy array
	:params mu1, mu2: mean of each poissonian, dtype = float
	:params A: relative amplitude of first poissonian, dtype = float
	"""
	poisson1 = Poisson(mu1)
	poisson2 = Poisson(mu2)
	return A * poisson1.pmf(x) + (1-A) * poisson2.pmf(x)

# def modified_Poisson(x, mu1, mu2, A):
# 	"""
# 	Sum of 2 poissonian distributions: `A1*exp{-mu1}*mu1^{x}/x! + A2*exp{-mu2}*mu2^{x}/x!`\n
# 	:params x_data_tuple: independent variables, dtype = tuple of numpy array
# 	:params mu1, mu2: mean of each poissonian, dtype = float
# 	:params A: relative amplitude of first poissonian, dtype = float
# 	"""
# 	poisson1 = Poisson(mu1)
# 	poisson2 = Poisson(mu2)
# 	return A * poisson1.pmf(x) + (1-A) * poisson2.pmf(x)

def My_Poisson(x, mu1):
	"""
	Poissonian distributions: `A*exp{-mu1}*mu1^{x}/x!`\n
	:params x_data_tuple: independent variables, dtype = tuple of numpy array
	:params mu1: mean poissonian, dtype = float
	:params A: amplitude of poissonian, dtype = float
	"""
	poisson1 = Poisson(mu1)
	return poisson1.pmf(x)

@njit(nopython=True)
def Lorentzian_peak(x, A, x0, gamma, offset):
	"""
	Lorentzian_peak(x, A, x0, sigma, offset)
	1D Lorentzian: A * 1/(1+(x-x0)^2/gamma^2)) + offset
	x: array
	A: amplitude. dtype = float
	x0: center. dtype = float
	gamma: width. dtype = float
	offset: offset. dtype = float
	"""
	f = A*(1/(1+(x-x0)**2/gamma**2)) + offset
	return f

@njit(nopython=True)
def double_Lorentzian_peak(x, A1, x01, gamma1, A2, x02, gamma2, offset):
	"""
	Lorentzian_peak(x, A, x_0, sigma, offset)
	1D Lorentzian: A * 1/(1+(x-x_0)^2/gamma^2)) + offset
	x: array
	A: amplitude. dtype = float
	x01, x02: centers. dtype = float
	gamma: width. dtype = float
	offset: offset. dtype = float
	"""
	f = A1*(1/(1+(x-x01)**2/gamma1**2)) + A2*(1/(1+(x-x02)**2/gamma2**2)) + offset
	return f


########################################################
############### Offline analysis #######################
########################################################
# @njit(nopython=True)
def prepare_data_plots(x_data, y_data):
	"""
	Returns x_data reduced to unique values and sorted, and mean(y_data(x_data)) with std dev
	x_data: array with independent variable
	y_data: array with dependent variable
	"""
	# List all unique values of x
	x_unique = list(set(x_data))
	y_means = []
	y_stds = []

	for x_val in x_unique:
		# Find the location of all unique x values
		positions = np.where(x_data==x_val)[0]
		# These are the y values corresponding to those x values
		try:
			y_values = y_data.iloc[positions]
		except:
			y_values = y_data[positions]
		# Compute mean and std
		y_means.append(np.mean(y_values))
		y_stds.append (np.std(y_values)/np.sqrt(len(y_values)))

	# Reorder data
	data = np.asarray(sorted(zip(x_unique,y_means,y_stds)))
	x_sorted = data[:,0]
	y_sorted = data[:,1]
	std_sorted = data[:,2]

	return x_sorted, y_sorted, std_sorted

def compute_T(TOF,isotope = 174):
	m = isotope*constants.physical_constants["atomic mass constant"]
	k_B = constants.physical_constants["Boltzmann constant"]
	return

def make_rois(size, centers, single_column = True):
	roi_corners = np.round(centers,0)-size//2
	tweezer_patches = []
	corners = np.round(roi_corners,0).astype(np.int16)
	if single_column: 
		# Find most common column
		col = mode(corners[:,0]).mode
		corners[:,0] = np.full(corners[:,0].shape,col)

	for i in range(centers.shape[0]):
		tweezer_patches.append(patches.Rectangle((corners[i,0]-1,corners[i,1]-1),size+1,size+1, linewidth=1, edgecolor='red', facecolor='none'))

	return corners, np.asarray(tweezer_patches)

def make_rectangular_rois(size_h, size_v, centers):
	roi_corners_x = np.round(centers[:,0],0)-size_h//2
	roi_corners_y = np.round(centers[:,1],0)-size_v//2

	tweezer_patches = []
	corners_x = np.round(roi_corners_x,0).astype(np.uint16)
	corners_y = np.round(roi_corners_y,0).astype(np.uint16)

	for i in range(centers.shape[0]):
		tweezer_patches.append(patches.Rectangle((corners_x[i]-1,corners_y[i]-1),size_h+1,size_v+1, linewidth=1, edgecolor='red', facecolor='none'))

	return corners_x, corners_y, np.asarray(tweezer_patches)


def make_circular_rois(r, centers):
	centers_round = np.round(centers,0).astype(np.uint16)

	tweezer_patches = []

	for i in range(centers.shape[0]):
		tweezer_patches.append(patches.Circle((centers[i,0],centers[i,1]), radius = r, linewidth=1, edgecolor='red', facecolor='none'))

	return centers_round, np.asarray(tweezer_patches)

def make_elliptic_rois(a, b, centers):
	centers_round = np.round(centers,0).astype(np.uint16)

	tweezer_patches = []

	for i in range(centers.shape[0]):
		tweezer_patches.append(patches.Ellipse((centers[i,0],centers[i,1]), width = a, height = b, linewidth=1, edgecolor='red', facecolor='none'))
	return centers_round, np.asarray(tweezer_patches)

def create_circular_mask(center, radius, shape):
	y, x = np.ogrid[:shape[0], :shape[1]]
	distance_squared = (x - center[0])**2 + (y - center[1])**2
	mask = distance_squared <= radius**2
	return mask


##############################################################
#################### File handling ###########################
##############################################################

# def import_shot_files(isotope, year, month, day, run, sequence_indexes):
# 	path = f'//ARQUS-CAM/Experiments/ytterbium{isotope}/{year}/{month}/{day}/{run}/shots/'
# 	all_files = os.listdir(path)
# 	files_to_analyse = []
# 	for file in all_files:
# 		for sequence_index in sequence_indexes:
# 			if file[12:15] == sequence_index:
# 				files_to_analyse.append(file)
# 	return files_to_analyse, path

# def get_everything_from_files(files_to_analyse,x_scan_parameter,label_scan_parameter, group, result,condition_param,condition_value):
# 	"""
# 	Useful for old analysis
# 	"""
# 	x_values = []
# 	label_values = []
# 	results = []
# 	for file in files_to_analyse:
# 		f = h5py.File(path+file, 'r')
# 		if condition_param is None or f['globals'].attrs[condition_param] == condition_value:
# 			x_values.append(f['globals'].attrs[x_scan_parameter])
# 			label_values.append(f['globals'].attrs[label_scan_parameter])
# 			results.append(f[group].attrs[result])
# 		f.close()
# 	x_uniques = np.asarray(list(set(x_values)))
# 	label_uniques= np.asarray(list(set(label_values)))
# 	return np.asarray(x_values), np.asarray(label_values), np.sort(x_uniques), np.sort(label_uniques),  np.asarray(results)

def get_shots_path(isotope: int, year: int, month: int, day: int, run: str):
	"""
	Build the string of the shotfiles path, depending on the given run info.
	If the path does not exist, an alternative path is returned.

	:params isotope: isotope mass number
	:type isotope: int
	:params year: year of measurement
	:type year: int
	:params month: month of measurement
	:type month: int
	:params day: day of measurement
	:type day: int
	:params run: name of the experimental run
	:type run: str
	:return: return a string containing the data path
	:rtype: str
	"""
	path = f'//ARQUS-CAM/Experiments/ytterbium{isotope}/{year}/{month}/{day}/{run}/shots/'
	# check if the path exists
	if not os.path.exists(path):
		# if not return path of second SSD
		path = f'//ARQUS-CAM/Experiments2/ytterbium{isotope}/{year}/{month}/{day}/{run}/shots/'

	return path


def import_shot_files(path, sequence_indexes):
	"""
	Build a list of shot file names to be analyzed, depending on given path and sequence indexes and returns the list with the path of the parent folder.
	path: path of the folder with shotfiles, dtype=str
	sequence_indexes: list of sequence indexes, dtype=[str]
	"""
	files_to_analyse = []
	for file in os.listdir(path):
		for sequence_index in sequence_indexes:
			if file[12:15] == sequence_index:
				files_to_analyse.append(file)
	return files_to_analyse

def shot_file_data_loading(filename: str, datafields: list[str], N_imgs: int, img_range: list[list[int]] = [ORCA_ROI_ROWS, ORCA_ROI_COLS]):
	"""
	Load data from shotfile, returning a list where the first element are all the images and the second element are all the other information from the shots, chosen in datafields.

	:params filename: complete path of the file 
	:type filename: str
	:params datafields: list of attributes to be loaded from the 'global' key in the shot file
	:type datafields: [str]
	:params N_imgs: number of images in each shot
	:type N_imgs: int
	:params img_range: list of indexes for the image, in form [[ROW1,ROW2],[COL1,COL2]]
	:type img_range: [[int]]
	"""	
	results = [[] for _ in datafields]
	temp = []
	images = []
	f = h5py.File(filename, 'r')
	try:
		for j in range(len(datafields)):
			results[j].append(f['globals'].attrs[datafields[j]])
		for img_n in range(N_imgs):
			temp.append(np.asarray(np.transpose(f[f'images/Orca/fluorescence {img_n+1}/frame']))[img_range[0][0]:img_range[0][1],img_range[1][0]:img_range[1][1]])
		images.append(np.asarray(temp))
	except:
			print(f'Problem with file {filename}')



	return np.asarray(images), np.asarray(results)

def parallel_data_loading(path: str, files: list[str], datafields: list[str],  N_imgs: int, img_range: list[list[int]], njobs: int = -1):
	"""
	Parallel loading of data from shot files, using the function shot_file_data_loading from above and joblib.

	:params path: complete path of the directory
	:type path: str
	:params files: list of files names
	:type files: [str]
	:params datafields: list of attributes to be loaded from the 'global' key in the shot file
	:type datafields: [str]
	:params N_imgs: number of images in each shot
	:type N_imgs: int
	:params img_range: list of indexes for the image, in form [[ROW1,ROW2],[COL1,COL2]]
	:type img_range: [[int]]
	:params njobs: jobs to be used by joblib
	:type njobs: int
	"""
	output = []
	func = delayed(shot_file_data_loading)
	data = Parallel(n_jobs=njobs)(func(path+f, datafields, N_imgs, img_range=img_range) for f in files)
	for el in data:
		temp=[]
		temp.append(el[0])
		for i in range(len(el[1])):
			temp.append(el[1][i])
		output.append(temp)
	return output

def data_splitting(files: list[str], data: list):
	"""
	Split the data loaded with the function parallel_data_loading in different lists. Returns n lists, where n is the number of loaded data fields.

	:params files: list of filenames, to know the total shots number
	:type files: [str]
	:params data: list of data
	:type data: []
	"""
	output = [[] for _ in data[0]]
	first_error = 0
	incr = 0
	for i in range(len(files)):
		try:
			for j in range(len(data[i])):
				output[j].append(data[i][j][0])
		except:
			if first_error == 0:
				first_error = i
			elif first_error == i-incr and incr == 1:
				print(f'No data from pos {first_error}/{len(files)-1}')
			elif first_error != i-incr:
				print(f'No data in pos {i}/{len(files)-1}')
			incr+=1
	return output

def get_keys_list_hdf5_file(filename):
	"""
	Returns a list of the available keys in the hdf5 filename file.
	filename: filename with absolute or relative path, dtype=str
	"""
	keys = list(h5py.File(filename).keys())
	h5py.File(filename).close()
	return keys

def write_single_hdf5_file(filename, key, value):
	"""
	Write a single variable value on a hdf5 file, with the corresponding key reference.
	filename: filename with absolute or relative path, dtype=str
	key: string key associated to the variable, dtype=str
	value: variable to be written
	"""
	with h5py.File(filename, 'w') as file:
		file[key] = value
		file.close()
	return

def write_hdf5_file(filename, keys, values):
	"""
	Write a list of variables on a hdf5 file, with the corresponding key references.
	Note 1: being in a list, the value to be written can be not homogeneous in type or dimension
	Note 2: at the moment, we are not using the group structure of the hdf5 file, but it's possible to use it.
	filename: filename with absolute or relative path, dtype=str
	key: list of string keys associated to the variables, dtype=[str]
	values: variables to be written, dtype=list
	"""
	if len(keys) != len(values):
		print('Keys number doesn\'t match variables number!')
		return
	with h5py.File(filename, 'w') as file:
		for i in range(len(keys)):
			file[keys[i]] = values[i]
		file.close()
	return

def add_note_hdf5_file(filename, keys, notes):
	"""
	Add notes to any key match registered in a hdf5 file.
	filename: filename with absolute or relative path, dtype=str
	key: list of strings referring to variables in a hdf5 file, dtype=[str]
	notes: list of strings of notes, dtype=[str]
	"""
	file_keys = get_keys_list_hdf5_file(filename)
	with h5py.File(filename, 'r+') as file:
		for i in range(len(keys)):
			if not keys[i] in file_keys:
				print(f'\'{keys[i]}\' is not available in {filename}')
				return
			file[keys[i]].attrs['notes'] = notes[i]
		file.close()
	return

def read_single_hdf5_file(filename, key):
	"""
	Extract the value corresponding to a record in a hdf5 file, associated to the key.
	filename: filename with absolute or relative path, dtype=str
	key: string key associated to the saved variable, dtype=str
	"""
	# if not (key in get_keys_list_hdf5_file(filename)):
	# 	print(f'The key is not present in the {filename}')
	# 	return
	with h5py.File(filename, 'r') as file:
		data = file[key][()]
		file.close()
	return data

def read_hdf5_file(filename, keys):
	"""
	Extract a list of dataset from a hdf5 file, each dataset associated to a key.
	Note 1: the returned data is a list since the type and dimension of the datasets can be not homogeneous.
	Note 2: at the moment, we are not using the group structure of the hdf5 file, but it's possible to use it.
	filename: filename with absolute or relative path, dtype=str
	key: list of string keys associated to the saved variables, dtype=[str]
	"""
	# for i in range(len(keys)):
	# 	if not keys[i] in get_keys_list_hdf5_file(filename):
	# 		print(f'\'{keys[i]}\' is not available in {filename}')
	# 		return
	data = []
	for i in range(len(keys)):
		data.append(read_single_hdf5_file(filename, keys[i]))
	return data




##############################################################
################### Fitting functions: #######################
##############################################################

def sum_of_1D_gaussians(xdata_tuple, *params):
	n = len(params) // 4  # Each Gaussian has 4 parameters: amp, x0, sigma_x, offset
	G = 0
	for i in range(n):
		amp, x0, sigma_x, offset = params[i * 4: ((i + 1) * 4)]
		amp = np.abs(amp)
		G += Gaussian(xdata_tuple, amp, x0, sigma_x, offset)
	return G


def sum_of_2D_gaussians(xdata_tuple, *params):
	n = len(params) // 6  # Each Gaussian has 5 parameters: amp, x0, y0, sigma_x, sigma_y
	G = 0
	for i in range(n):
		amp, x0, y0, sigma_x, sigma_y, offset = params[i * 6: ((i + 1) * 6)]
		G += Gaussian_2D(xdata_tuple, amp, x0, y0, sigma_x, sigma_y, offset)
	return G.ravel()

def tweezers_multigauss_fit(img,N_tweezers, debug = False):
	flattened = img.flatten() # flattens goes rows first
	peaks = signal.find_peaks(flattened,distance=flattened.shape[0]/(N_tweezers+1))

	for i in range(5):
		if peaks[0].shape[0] != N_tweezers:
			peaks = signal.find_peaks(flattened,distance=flattened.shape[0]/(N_tweezers+i))
	if debug:
		fig_debug, ax_debug = plt.subplots(1,3,figsize=(25,10))
		ax_debug[0].plot(flattened,label='data')
		ax_debug[0].plot(peaks[0],flattened[peaks[0]],linestyle='',marker='*',label='detected peaks',color='r')
		ax_debug[0].legend(bbox_to_anchor=([1,1]))
		ax_debug[0].set_xlabel('pixels')
		ax_debug[0].set_ylabel('counts')
		fig_debug.suptitle('debug image')

	if peaks[0].shape[0] != N_tweezers:
		plt.show()
		raise Exception(f'{peaks[0].shape[0]} peaks found in the image but {N_tweezers} tweezers are expected!')
	peaks_rows = peaks[0]/img.shape[1]
	peaks_col = ((peaks_rows-np.floor(peaks_rows))*img.shape[1])
	peaks_pos = np.zeros((N_tweezers,2))
	peaks_pos[:,1] = peaks_rows
	peaks_pos[:,0] = peaks_col
	if debug:
		im = ax_debug[1].imshow(img,cmap='Purples')
		ax_debug[1].set_title('data')
		fig_debug.colorbar(im, ax=ax_debug[1])
		# ax_debug[1].plot(peaks_pos[:,0],peaks_pos[:,1],marker='*',linestyle='',color='r',markersize=4)
	# Guess parameters of the N gaussians
	# Parameters are: amplitude, center x, center y, sigma x, sigma y, offset
	params = np.asarray([np.max(img)-np.min(img), 10, 10, 1, 1, np.min(img)] * N_tweezers)
	# Fix centers guess with the maxima positions
	for i in range(N_tweezers):
		params[1+(i*6)] = peaks_pos[i,0]
		params[2+(i*6)] = peaks_pos[i,1]
	x = np.arange(img.shape[1])
	y = np.arange(img.shape[0])
	x, y = np.meshgrid(x, y)
	p_opt, cov = opt.curve_fit(sum_of_2D_gaussians, (x, y), img.ravel(), p0 = params)
	data_fitted = sum_of_2D_gaussians((x, y), *p_opt).reshape(y.shape)
	fit_res = p_opt.reshape(N_tweezers,6)
	errs = np.sqrt(np.diag(cov))
	fit_errs = errs.reshape(N_tweezers,6)
	if debug:
		fit = ax_debug[2].imshow(data_fitted,cmap='Purples')
		ax_debug[2].set_title('fit')
		fig_debug.colorbar(fit, ax=ax_debug[2])
		ax_debug[1].plot(fit_res[:,1],fit_res[:,2],marker='*',linestyle='',color='r',markersize=3)
		plt.show()
	return fit_res, fit_errs

##############################################################
###################### Other  ################################
##############################################################

def find_threshold(sums, sequence_indexes, x_lim = 150, y_lim = 1e-3, logscale=False, poisson=True, show_plot = True):
	n_bins = round(np.max(sums)-np.min(sums))
	counts, bins = np.histogram(sums, bins = n_bins, density=True)

	if poisson:
		x = bins[:-1]
		x_fit = bins[:-1]
		fit_function = double_Poisson
		param_start = [1, 0.6*np.max(bins), 0.5]
		param_bounds = ([0, 0, 0.01],[np.max(bins), np.max(bins), 1])
	else:
		x = np.linspace(-1.2*np.max(bins),1.2*np.max(bins),400) #x range changed to take care in the normalization integral of the fitted dark gaussian having negative x
		x_fit = bins[:-1]+(bins[1]-bins[0])/2
		fit_function = normalized_double_Gaussian
		param_start = [1, 1, 0.25*np.max(bins), 10, 0.3]
		param_bounds = ([np.min(bins), 0, np.min(bins), 0, 0],[np.max(bins), np.inf, np.max(bins), np.inf, 1])


	p_opt, p_cov = opt.curve_fit(fit_function, x_fit, counts/(np.sum(counts) * np.diff(bins)),
							  p0 = param_start, bounds = param_bounds)
	if poisson:
		u = p_opt[2]
		total_fit = double_Poisson(np.round(x,0), *p_opt)
		errs = np.sqrt(np.diag(p_cov))

		if p_opt[0] < p_opt[1]:
			if show_plot: print('Tweezer filling P =',1-u)
			dark_fit = u*Poisson(p_opt[0]).pmf(np.round(x,0))
			atom_fit = (1-u)*Poisson(p_opt[1]).pmf(np.round(x,0))
		else:
			if show_plot: print('Tweezer filling P =',u)
			dark_fit = u*Poisson(p_opt[1]).pmf(np.round(x,0))
			atom_fit = (1-u)*Poisson(p_opt[0]).pmf(np.round(x,0))

		if p_opt[0] < p_opt[1]:
			arg_atom = p_opt[1]
			arg_dark = p_opt[0]
		else:
			arg_atom = p_opt[0]
			arg_dark = p_opt[1]

	else:
		u = p_opt[4]
		total_fit = normalized_double_Gaussian(x, *p_opt)
		errs = np.sqrt(np.diag(p_cov))

		if p_opt[0] < p_opt[2]:
			if show_plot:  print('Tweezer filling P =',1-u)
			dark_fit = u*normalized_Gaussian(x, *p_opt[0:2])
			atom_fit = (1-u)*normalized_Gaussian(x, *p_opt[2:4])
		else:
			if show_plot:  print('Tweezer filling P =',u)
			dark_fit =u*normalized_Gaussian(x, *p_opt[2:4])
			atom_fit = (1-u)*normalized_Gaussian(x, *p_opt[0:2])

		if p_opt[0] < p_opt[2]:
			arg_atom = tuple(e for e in p_opt[2:4])
			arg_dark = tuple(e for e in p_opt[0:2])
		else:
			arg_atom = tuple(e for e in p_opt[0:2])
			arg_dark = tuple(e for e in p_opt[2:4])

	th_list = np.linspace(3, np.max(bins)*0.5, 100)
	integral = []

	if poisson:
		x_th = arg_atom/np.log(1+arg_atom/arg_dark)

	if not poisson:
		for th in th_list:
			integral.append(quad(normalized_Gaussian, -1.2*np.max(bins), th, args=arg_atom)[0] + quad(normalized_Gaussian, th, 1.2*np.max(bins) , args=arg_dark)[0])

		x_th = th_list[np.argmin(integral)]

	if 0.1 < u < 0.9:
		D = 1 - (np.trapz(atom_fit[x<x_th],x[x<x_th])+np.trapz(dark_fit[x>=x_th],x[x>=x_th]))   #New name for fidelity (Distinguishablility)
	else:
		D = 0
		print('Problem in the histogram fit')

	dark_bins = []
	for i in range(len(bins)):
		if bins[i] < x_th:
			dark_bins.append(bins[i]-0.5)
	atom_bins = []
	for i in range(len(bins)):
		if bins[i] >= x_th:
			atom_bins.append(bins[i]-0.5)

	atom_counts = counts[len(dark_bins):]
	dark_counts = counts[:len(dark_bins)]
	dark_bins += [atom_bins[0]]
	if show_plot:
		plt.hist(dark_bins[:-1], dark_bins, weights=dark_counts/(np.sum(counts) * np.diff(dark_bins)),color='grey',ls = '-')
		plt.hist(atom_bins[:-1], atom_bins, weights=atom_counts/(np.sum(counts) * np.diff(atom_bins)),color='mediumblue',ls = '-')
		if poisson:
			plt.plot(x, total_fit,label = f'atoms:\n$ \\langle \\varphi \\rangle $ = {np.round(arg_atom,1)}\n$ \\sigma_\\varphi $ = {np.round(np.sqrt(arg_atom),1)}\ndark:\n$ \\langle \\varphi \\rangle $ = {np.round(arg_dark,1)}\n$\\sigma_\\phi $ = {np.round(np.sqrt(arg_dark),1)}')
		else:
			plt.plot(x, total_fit,label = f'atoms:\n$ \\langle \\varphi \\rangle $ = {np.round(arg_atom[0],1)}\n$ \\sigma_\\varphi $ = {np.round(arg_atom[1],1)}\ndark:\n$ \\langle \\varphi \\rangle $ = {np.round(arg_dark[0],1)}\n$\\sigma_\\phi $ = {np.round(arg_dark[1],1)}')
		plt.xlim(-0.5,x_lim)
		plt.title(f'Sequence: {sequence_indexes}\nD: {np.round(D*100,3)}%\n Threshold: {round(x_th, 2)}',fontsize = 17)
		plt.axvline(x_th)
		plt.xlabel('photons')
		plt.ylabel('relative frequency')
		if logscale:
			plt.yscale('log')
			plt.ylim(y_lim,0.5)

		plt.legend(bbox_to_anchor=([1,1]),fontsize = 17)
		plt.show()

	return x_th, D, p_opt, errs


def fit_histogram(sums,sequence_indexes, x_lim = 150, y_lim = 1e-3, logscale=False, poisson=True, axvline = False):
	"""
	Fit a single peak of an histogram
	"""
	n_bins = round(np.max(sums)-np.min(sums))
	counts, bins = np.histogram(sums, bins = n_bins, density=True)

	if poisson:
		x = bins[:-1]
		x_fit = bins[:-1]
		fit_function = My_Poisson
		param_start = [0.5*np.max(bins)]
		param_bounds = (0,np.max(bins))
	else:
		x = np.linspace(-1.2*np.max(bins),1.2*np.max(bins),400) #x range changed to take care in the normalization integral of the fitted dark gaussian having negative x
		x_fit = bins[:-1]+(bins[1]-bins[0])/2
		fit_function = normalized_Gaussian
		param_start = [0.5*np.max(bins), np.std(bins)]
		param_bounds = ([0, 0.1],[np.max(bins),np.max(bins)])
	p_opt, p_cov = opt.curve_fit(fit_function, x_fit, counts/(np.sum(counts) * np.diff(bins)),
							  p0 = param_start)#, bounds = param_bounds)
	if poisson:
		total_fit = My_Poisson(np.round(x,0), *p_opt)
		errs = np.sqrt(np.diag(p_cov))
	else:
		total_fit = normalized_Gaussian(x, *p_opt)
		errs = np.sqrt(np.diag(p_cov))
	plt.hist(bins[:-1], bins, weights=counts/(np.sum(counts) * np.diff(bins)),color='mediumblue',ls = '-')
	if poisson:
		plt.plot(x, total_fit,label = f'Poissonian:\n$\\mu=${np.round(p_opt[0],2)}$\\pm${np.round(errs[0],2)}')
	else:
		plt.plot(x, total_fit,label = f'Gaussian:\n$x_0=${np.round(p_opt[0],2)}$\\pm${np.round(errs[0],2)}\n$\\sigma=${np.round(p_opt[1],2)}$\\pm${np.round(errs[1],2)}')
	plt.xlim(-0.5,x_lim)
	plt.title(f'Sequence: {sequence_indexes}',fontsize = 17)
	plt.xlabel('photons')
	plt.ylabel('relative frequency')
	if logscale:
		plt.yscale('log')
		plt.ylim(y_lim,0.5)
	if axvline:  
		plt.axvline(axvline,linestyle='-',color='gray',label = f'{axvline}')
	plt.legend(bbox_to_anchor=([1,1]),fontsize = 17)
	plt.show()

	return p_opt, errs


def compute_survival_P (x_data, y_data):
	# List all unique x values
	x_unique = list(set(x_data))
	y_all = []
	for x_val in x_unique:
		# Find the location of all unique x values
		positions = np.where(x_data==x_val)[0]
		# For each tweezer, count how many times an atom was there
		counter=np.sum(y_data[:,positions,:],axis = 1)
		# For each tweezer, get the survival probability by diving N times the atom is in the second image/N times it is in the 1st
		survival_P = counter[:,1]/counter[:,0]
		y_all.append(survival_P)
	y_all = np.asarray(y_all)
	x_unique = np.asarray(x_unique)
	inds = x_unique.argsort()
	y_sorted = y_all[inds,:]
	x_sorted = x_unique[inds]
	return x_sorted, y_sorted

def progressbar(current_value,total_value,bar_lengh,progress_char):
	"""
	current: current value of advancement, dtype = int
	total_value: total value of advancement, dtype = int
	bar_length: number of character corresponding to the full bar, dtype = int
	progress_char: filling character of the loading bar, dtype = str
	"""
	percentage = np.round(current_value/total_value,4)										 				# Percent Completed Calculation
	progress = int((bar_lengh * current_value ) / total_value)									   		# Progress Done Calculation
	loadbar = 'Progress: [{:{len}}] {:2.2%}'.format(progress*progress_char,percentage,len = bar_lengh)		# Progress Bar String
	print(loadbar, end='\r')																		 		# Progress Bar Output

def plot_histogram(sums, y_scale = 'log', y_lim = 1e-4, x_lim = 80, thresholds = None):
	n_bins = round(np.max(sums)-np.min(sums))
	counts, bins = np.histogram(sums, bins = n_bins, density=True)
	fig, ax = plt.subplots(1,1,figsize = (8,6))
	ax.hist(bins[:-1],bins,weights=counts,color='slateblue',histtype = 'bar',alpha = 0.7,edgecolor = 'navy')
	print()
	if y_scale == 'log': ax.set_yscale('log')
	ax.set_ylim(y_lim,)
	ax.set_xlim(-1,x_lim)
	ax.set_xlabel('Collected photons')
	ax.set_ylabel('Relative frequency')
	if thresholds is not None:
		for th in thresholds:
			ax.axvline(th,linewidth = 3,label=f'{np.round(th,1)}',color = 'tab:grey')
		ax.legend()
	plt.show()
	return 

###########################################################################################
############## Count atoms and losses + other fast imaging analysis ########################
############################################################################################
def count_losses(N1,N2):
	"""
	N1: Atoms in the first image
	N2: atoms in the second image
	Returns:
	P_loss (float): loss probaility. 
	errorbars (2,): which are the lower and upper errobars 
	"""
	N_survived = N1*N2
	P_loss = 1-(np.sum(N_survived)/np.sum(N1))
	if P_loss == 0:
		print(f'Losses limited by statistics')
		P_loss = 1/(np.sum(N1)+1)
	conf_levels = binom_conf_interval(np.sum(N_survived),np.sum(N1), interval = 'jeffreys')
	# conf_levels[0]([1]) is the lower(upper) limit to the survival probability
	P_min = 1-conf_levels[1]
	P_max = 1-conf_levels[0]
	errorbars = np.asarray([(P_loss-P_min), (P_max-P_loss)])
	return P_loss, errorbars

def count_atoms (sums,thresholds):
	N_atoms= np.zeros(sums.shape)
	if N_atoms.shape[0] != thresholds.shape[0]:
		print('Error!!')
		return
	# Shape 0 is the N of successive images
	for i in range(N_atoms.shape[0]):
		N_atoms[i,sums[i]>=thresholds[i]] = 1
	return N_atoms

def real_t_ill(t_ill,pulse_dur = 400e-9):
	"""
	Compute the real illumination time considering the alternating pulses
	"""
	total_pulse_lenght = pulse_dur*2+100e-9
	on_time = 2*pulse_dur/total_pulse_lenght
	return np.ceil((t_ill)/(total_pulse_lenght))*total_pulse_lenght*on_time # s # this time is calculated considering what our pulsed imaging is actually doing
	
def fit_histograms(sums, image = 1, show_plot = False):
	"""
	Find the tweezer-by-tweezer center of each histogram
	
	:params sums: shape (N images, N shots, N tweezers)
	"""
	hist_centers = []
	hist_centers_err = []
	for tweezer in range(sums.shape[-1]):
		x_th, F, fit_params,fit_erss = find_threshold(sums[image-1,:,tweezer],f'Image {image}',x_lim = 80,
														 logscale=True, poisson = True, y_lim = 2e-5,show_plot=show_plot)
		hist_centers.append(np.abs(fit_params[1]-fit_params[0]))
		hist_centers_err.append(np.sqrt(fit_erss[1]**2+fit_erss[0]**2))
	return np.array([np.asarray(hist_centers), np.asarray(hist_centers_err)])

def find_image_th(sums,image=2,show_plot=False, poisson = True):
	ths = []
	for tweezer in range(sums.shape[-1]):
		x_th,_,_,_= find_threshold(sums[image-1,:,tweezer],f'Tweezer:{tweezer+1}',x_lim = 80,
														 logscale=True, poisson = poisson, y_lim = 2e-5,show_plot=show_plot)
		ths.append(x_th)
	ths = np.asarray(ths)
	return ths

def plot_2D_histogram(sums_first,sums_second,c_x_opt,histogram_lim = 5e-4,xlim = 70, ylim = 70, alpha = 0.5, nice_colouring = False):
	x_vals = sums_first.flatten()
	y_vals = sums_second.flatten()
	xy = np.vstack([x_vals, y_vals])
	if nice_colouring:
		density = gaussian_kde(xy)(xy)
		custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom_blue", ["lavender", "slateblue"])
	fig = plt.figure(figsize=(8, 8))
	gs = fig.add_gridspec(2, 2, width_ratios=(3, 1), height_ratios=(1, 3), left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.05, hspace=0.05)
	ax = fig.add_subplot(gs[1, 0])
	ax_top = fig.add_subplot(gs[0, 0])
	ax_right = fig.add_subplot(gs[1, 1])
	if nice_colouring: sc = ax.scatter(x_vals, y_vals, c=density, cmap=custom_cmap, s=20, alpha=alpha, vmin=np.min(density), vmax=np.max(density)/10)
	else: sc = ax.scatter(x_vals, y_vals, c="slateblue", s=32, alpha=0.02, edgecolors='none', linewidths=0.)
	try:
		ax.axvline(x=c_x_opt, alpha=0.5, color='black', linestyle='--')
		ax.axhline(y=c_x_opt, alpha=0.5, color='black', linestyle='--')
	except:
		for th in c_x_opt:
			ax.axvline(x=th, alpha=0.5, color='black', linestyle='--')
			ax.axhline(y=th, alpha=0.5, color='black', linestyle='--')
	
	ax.set_xlabel("First image photons")
	ax.set_ylabel("Second image photons")
	ax.tick_params(axis='both', direction='in', length=6, width=2)


	ax_top.hist(x_vals, bins=np.max(x_vals), color='slateblue', histtype = 'bar',alpha = 0.7,edgecolor = 'navy', density=True)
	ax_right.hist(y_vals, bins=np.max(y_vals), color='slateblue', orientation='horizontal', histtype = 'bar',alpha = 0.7,edgecolor = 'navy', density=True)
	try:
		ax_top.axvline(x=c_x_opt, alpha=0.5, color='black', linestyle='--')
		ax_right.axhline(c_x_opt, alpha=0.5, color='black', linestyle='--')
	except:
		for th in c_x_opt:
			ax_top.axvline(x=th, alpha=0.5, color='black', linestyle='--')
			ax_right.axhline(y=th, alpha=0.5, color='black', linestyle='--')

	ax_top.tick_params(axis='both', direction='in', length=6, width=2)
	ax_right.tick_params(axis='both', direction='in', length=6, width=2)
	ax_top.tick_params(which='minor', bottom=False, left=False, right=False)
	ax_right.tick_params(which='minor', bottom=False, left=False, right=False,top = False)

	ax_top.set_ylabel("Frequency",fontsize = 20)
	ax_right.set_xlabel("Frequency",fontsize = 20)

	ax_top.set_yscale('log')
	ax_right.set_xscale('log')

	ax_top.set_xticklabels([])
	ax_right.set_yticklabels([])
	ax_top.set_ylim(histogram_lim,)
	ax_top.set_xlim(-2,xlim)
	ax_right.set_xlim(histogram_lim,)
	ax_right.set_ylim(-2,ylim)
	ax.set_ylim(-2,ylim)
	ax.set_xlim(-2,xlim)
	fig.tight_layout()
	ax_right.set_xticks([1e-3,1e-1])
	ax_top.set_yticks([1e-3,1e-1])
	plt.show()



# def count_atoms_2(sums,thresholds):
# 	N_atoms=np.zeros(sums.shape)
# 	if N_atoms.shape[0] != thresholds.shape[0]:
# 		print('Shapes mismatch')
# 		return
# 	#Shape 0 is the N of successive images
# 	#Shape -1 is the tweezer index

# 	for i in range(N_atoms.shape[0]):
# 		for j in range (N_atoms.shape[-1]):
# 			N_atoms[i,sums[i,:,:,:,j]>=thresholds[i,:,j]] = 1
# 	return N_atoms


def azimuthal_average(x,y,z,n_bins=15):
    x0,y0 = np.mean(x), np.mean(y)
    
    r = np.hypot(x-x0,y-y0) 
    r_bins = np.linspace(0,r.max(),n_bins+1)
    r_bins_center = 0.5*(r_bins[:-1]+r_bins[1:])
    
    z_avg = np.zeros(n_bins)
    
    for i in range (n_bins):
        mask = (r>=r_bins[i]) & (r<=r_bins[i+1])
        if np.any(mask):
            z_avg[i] = np.mean(z[mask])
        else:
            z_avg[i]=np.nan
    
    return z_avg,r_bins_center





def AtomPositionHistogram(positionsX, positionsY, bins =7, binLimit = 10e-6, title = "Position after TOF"):
    
    x_edges = np.linspace(-binLimit, binLimit, bins + 1) 
    y_edges = np.linspace(-binLimit, binLimit, bins + 1) 
    
    plt.figure(figsize=(6,5))
    plt.hist2d(positionsX*1e6, positionsY*1e6, bins=[x_edges*1e6, y_edges*1e6], density=True, cmap='Blues')
    plt.xlabel(r"$\mu$m")
    plt.ylabel(r"$\mu$m")
    plt.title(title)
    plt.colorbar(label='Density')
    plt.xlim(-binLimit*1e6, binLimit*1e6)
    plt.ylim(-binLimit*1e6, binLimit*1e6)
    plt.show()
    
    
def AtomPositionAzimuthalAverage(positionX, positionY, bins = 5,plot = True, initial_guess =None):
    
    positionX = positionX*1e6
    positionY = positionY *1e6
    
    histogramValues, xedges, yedges = np.histogram2d(positionX, positionY, bins=(10,10))

    xValues, yValues, density = [],[],[]

    for y in range(histogramValues.shape[0]):
        for x in range(histogramValues.shape[1]):
            xValues.append(xedges[x])
            yValues.append(yedges[y])
            density.append(histogramValues[x][y])

    xValues = np.asarray(xValues)
    yValues = np.asarray(yValues)
    density = np.asarray(density)
    
    azimuthalAverageDensity, azimuthalPositions = azimuthal_average(xValues,yValues,density,n_bins=bins)
    lower_bounds = [0.95,  -0.01, -np.inf]
    upper_bounds = [1.05, 0.01,  np.inf]
    parameters, covariance = curve_fit(gaussian,azimuthalPositions , azimuthalAverageDensity/np.max(azimuthalAverageDensity), p0 = initial_guess,bounds=(lower_bounds, upper_bounds))

    
    if plot:
        azimuthalPositionsFit = np.linspace(0,np.max(azimuthalPositions),1000)
        plt.figure()
        plt.scatter(azimuthalPositions,azimuthalAverageDensity/np.max(azimuthalAverageDensity),marker='o',s=150,edgecolors='black')
        plt.plot(azimuthalPositionsFit, gaussian(azimuthalPositionsFit, *parameters), 'r-', label='Fitted Curve', linewidth=2,color = 'red')
        plt.xlabel(r'$\mu$m')
        plt.ylabel('a.u.')


        plt.title('Azimuthal average')
        plt.legend()
        plt.show()

    
    return parameters[1] * 1e-6, parameters[2]*1e-6



def gaussian(x, A, mu, sigma):
    """
    1D Gaussian function.
    A     : amplitude
    mu    : mean (center)
    sigma : standard deviation
    offset: baseline offset
    """
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) 




    
 








