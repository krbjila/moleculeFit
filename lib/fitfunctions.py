import numpy as np
import mpmath
mpmath.dps = 20

from scipy.optimize import least_squares
import defaults

from matplotlib import pyplot as plt
import numpy.random as random

from skimage.feature import peak_local_max
from scipy import ndimage as ndimage

def subtract_gradient(od):
    (dy, dx) = np.shape(od)
    X2D,Y2D = np.meshgrid(np.arange(dx),np.arange(dy))
    A = np.matrix(np.column_stack((X2D.ravel(),Y2D.ravel(),np.ones(dx*dy))))
    B = od.flatten()
    C = np.dot((A.T * A).I * A.T, B).flatten()
    bg=np.reshape(C*A.T, (dy,dx))
    return np.asarray(od - bg)

def gauss_fit(p,r,y):
	(xx,yy) = np.meshgrid(r[0], r[1])
	[offset, peak, xc, yc, sigx, sigy] = p
	return np.ravel(offset + peak*np.exp(-0.5*(xx-xc)**2.0/sigx**2.0 - 0.5*(yy-yc)**2.0/sigy**2.0) - y)

def gauss_grad_fit(p,r,y):
	(xx,yy) = np.meshgrid(r[0], r[1])
	[offset, peak, xc, yc, sigx, sigy, gradx, grady] = p
	return np.ravel(offset + gradx*(xx-xc) + grady*(yy-yc) + peak*np.exp(-0.5*(xx-xc)**2.0/sigx**2.0 - 0.5*(yy-yc)**2.0/sigy**2.0) - y)

npli = np.frompyfunc(mpmath.fp.polylog, 2, 1)

# See https://arxiv.org/pdf/0801.2500.pdf
def f(x):
	return (1 + x)*np.log(1 + x)/x

def fermi3d(p,r,y):
	(xx,yy) = np.meshgrid(r[0], r[1])
	[offset, peak, xc, yc, sigx, sigy, q] = p
	arg = -np.exp(q - f(np.exp(q)) * (0.5*np.square((xx.ravel() - xc)/sigx) + 0.5*np.square((yy.ravel() - yc)/sigy)))
	return np.array(offset + peak*npli(2, arg) / npli(2, -np.exp(q)) - y.ravel(), dtype=np.float)

def fermi2d(p,r,y):
	[offset, peak, xc, sigx, q] = p
	arg = -np.exp(q - f(np.exp(q))*0.5*np.square((r.ravel() - xc)/sigx) )
	return np.array(offset + peak*npli(3/2, arg) / npli(3/2, -np.exp(q)) - y.ravel(), dtype=np.float)

def TTF2d(q):
	return 1.0/np.sqrt(-2.0*npli(2,-np.exp(q)))

def TTF3d(q):
	return np.power(-6.0*npli(3, -np.exp(q)), -1.0/3.0)

def fitter(fname, data, bounds, xaxis, yaxis, rp, binning):
	(xmin, xmax, ymin, ymax) = bounds
	(width, height) = (xmax - xmin, ymax - ymin)

	if fname == "Gaussian":
		guess = [0, 0.5, rp["xc"], rp["yc"], 0.2*width, 0.2*height]
		upper_bounds = [defaults.max_od, defaults.max_od, xmax, ymax, width, height]
		lower_bounds = [-defaults.max_od, 0, xmin, ymin, 0, 0]

		res = least_squares(gauss_fit, guess, args=([xaxis, yaxis], data), bounds=(lower_bounds, upper_bounds))
		if not res.success:
			print("Warning: fit did not converge.")
		fits = {
			"f": fname,
			"offset": res.x[0],
			"peak": res.x[1],
			"xc": res.x[2],
			"yc": res.x[3],
			"sigx": res.x[4],
			"sigy": res.x[5]
		}

		fitted = gauss_fit(res.x, [xaxis, yaxis], 0)
		fitted = np.reshape(fitted, (height, width))

		fitted_x = np.sum(fitted, axis=0) * defaults.od_to_number * binning**2.0 / height
		fitted_y = np.sum(fitted, axis=1) * defaults.od_to_number * binning**2.0  / width

		return (fits, fitted_x, fitted_y)

	if fname == "Gaussian w/ Gradient":
		od_no_bg = subtract_gradient(data)
		blur = ndimage.gaussian_filter(od_no_bg,5,mode='constant')

		# plt.figure()
		# plt.imshow(blur)
		# plt.show()
		
		upper_bounds = [defaults.max_od, defaults.max_od, xmax, ymax, width, height, 0.000375, 0.000375]
		lower_bounds = [-defaults.max_od, 0, xmin, ymin, 0, 0, -0.000375, -0.000375]
		guess = [0, 0.5, rp["xc"], rp["yc"], 0.2*width, 0.2*height, 0.0, 0.0]

		pks=peak_local_max(blur, min_distance=20,exclude_border=2, num_peaks=3)
		guesses = [guess]
		for pk in pks:
			yc = pk[0]
			xc = pk[1]
			peak = data[yc, xc]
			offset = np.mean(data)
			if peak > 0:
				(sigx, sigy) = 15, 15
				guess = [offset, peak, xmin+xc, ymin+yc, sigx, sigy, 0.0, 0.0]
				guesses.append(guess)

		best_fit = None
		best_guess = np.inf		

		for guess in guesses:
			print(upper_bounds)
			print(lower_bounds)
			print(guess)
			try:
				res = least_squares(gauss_grad_fit, guess, args=([xaxis, yaxis], data), bounds=(lower_bounds, upper_bounds))
				print(res.cost)
				print(res.x)
				if not res.success:
					print("Warning: fit did not converge.")
				elif res.cost < best_guess:
					best_guess = res.cost
					best_fit = res
			except ValueError as e:
				print(e)

		res = best_fit
		fits = {
			"f": fname,
			"offset": res.x[0],
			"peak": res.x[1],
			"xc": res.x[2],
			"yc": res.x[3],
			"sigx": res.x[4],
			"sigy": res.x[5],
			"gradx": res.x[6],
			"grady": res.x[7],
		}

		fitted = gauss_grad_fit(res.x, [xaxis, yaxis], 0)
		fitted = np.reshape(fitted, (height, width))

		fitted_x = np.sum(fitted, axis=0) * defaults.od_to_number * binning**2.0 / height
		fitted_y = np.sum(fitted, axis=1) * defaults.od_to_number * binning**2.0  / width

		return (fits, fitted_x, fitted_y)

	elif fname == "Fermi 2D":
		(fit_gauss, f_x, f_y) = fitter("Gaussian", data, bounds, xaxis, yaxis, rp, binning)
		guess = [fit_gauss["offset"], fit_gauss["peak"], fit_gauss["xc"], fit_gauss["sigx"], 0] # q=0 is T/TF=0.78
		upper_bounds = [fit_gauss["offset"]+0.1, defaults.max_od, xmax, width, 50] # q=50 is T/TF=0.02
		lower_bounds = [fit_gauss["offset"]-0.1, -0.1, xmin, 0, -5] # q=-5 is T/TF=8

		# Integrate out vertical dimension
		data_int = np.sum(data, axis=0) / height

		res = least_squares(fermi2d, guess, args=(xaxis, data_int), bounds=(lower_bounds, upper_bounds))
		if not res.success:
			print "Warning: fit did not converge."

		fits = {
			"f": fname,
			"offset": res.x[0],
			"peak": res.x[1],
			"xc": res.x[2],
			"sigx": res.x[3]/np.sqrt(f(np.exp(res.x[4]))),
			"TTF": TTF2d(res.x[4])
		}

		fitted = fermi2d(res.x, xaxis, np.array([0]))
		fitted_x = fitted * defaults.od_to_number * binning**2.0 

		fits.update(
			{
				"peakGauss": fit_gauss["peak"],
				"sigxGauss": fit_gauss["sigx"],
				"sigyGauss": fit_gauss["sigy"],
				"ycGauss": fit_gauss["yc"]
			})
		return (fits, fitted_x, [])

	elif fname == "Fermi 3D":
		(fit_gauss, f_x, f_y) = fitter("Gaussian", data, bounds, xaxis, yaxis, rp, binning)
		guess = [fit_gauss["offset"], fit_gauss["peak"], fit_gauss["xc"], fit_gauss["yc"], fit_gauss["sigx"], fit_gauss["sigy"], 0] # q=0 is T/TF=0.57
		upper_bounds = [fit_gauss["offset"]+0.1, fit_gauss["peak"]+0.5, xmax, ymax, width, height, 50] # q=50 is T/TF=0.02
		lower_bounds = [fit_gauss["offset"]-0.1, fit_gauss["peak"]-0.5, xmin, ymin, 0, 0, -6] # q=-6 is T/TF=4

		res = least_squares(fermi3d, guess, args=([xaxis, yaxis], data), bounds=(lower_bounds, upper_bounds))
		if not res.success:
			print "Warning: fit did not converge."

		fits = {
			"f": fname,
			"offset": res.x[0],
			"peak": res.x[1],
			"xc": res.x[2],
			"yc": res.x[3],
			"sigx": res.x[4]/np.sqrt(f(np.exp(res.x[6]))),
			"sigy": res.x[5]/np.sqrt(f(np.exp(res.x[6]))),
			"TTF": TTF3d(res.x[6])
		}

		fitted = fermi3d(res.x, [xaxis, yaxis], np.array([0]))
		fitted = np.reshape(fitted, (height, width))

		fitted_x = np.sum(fitted, axis=0) * defaults.od_to_number * binning**2.0 / height
		fitted_y = np.sum(fitted, axis=1) * defaults.od_to_number * binning**2.0 / width

		fits.update(
			{
				"peakGauss": fit_gauss["peak"],
				"sigxGauss": fit_gauss["sigx"],
				"sigyGauss": fit_gauss["sigy"]
			})
		return (fits, fitted_x, fitted_y)

