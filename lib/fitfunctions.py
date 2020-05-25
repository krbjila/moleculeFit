import numpy as np
import mpmath
mpmath.dps = 20

from scipy.optimize import least_squares
import defaults

from matplotlib import pyplot as plt

def gauss_fit(p,r,y):
	(xx,yy) = np.meshgrid(r[0], r[1])
	[offset, peak, xc, yc, sigx, sigy] = p
	return np.ravel(offset + peak*np.exp(-0.5*(xx-xc)**2.0/sigx**2.0 - 0.5*(yy-yc)**2.0/sigy**2.0) - y)

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

def fitter(fname, data, bounds, xaxis, yaxis, rp):
	(xmin, xmax, ymin, ymax) = bounds
	(width, height) = (xmax - xmin, ymax - ymin)

	if fname == "Gaussian":
		guess = [0, 0.5, rp["xc"], rp["yc"], 0.2*width, 0.2*height]
		upper_bounds = [defaults.max_od, defaults.max_od, xmax, ymax, width, height]
		lower_bounds = [-defaults.max_od, 0, xmin, ymin, 0, 0]

		res = least_squares(gauss_fit, guess, args=([xaxis, yaxis], data), bounds=(lower_bounds, upper_bounds))
		if not res.success:
			print "Warning: fit did not converge."

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

		fitted_x = np.sum(fitted, axis=0) * defaults.od_to_number / height
		fitted_y = np.sum(fitted, axis=1) * defaults.od_to_number / width

		return (fits, fitted_x, fitted_y)

	elif fname == "Fermi 2D":
		(fit_gauss, f_x, f_y) = fitter("Gaussian", data, bounds, xaxis, yaxis, rp)
		guess = [fit_gauss["offset"], fit_gauss["peak"], fit_gauss["xc"], fit_gauss["sigx"], 0] # q=0 is T/TF=0.78
		upper_bounds = [fit_gauss["offset"]+0.1, defaults.max_od, xmax, width, 50] # q=50 is T/TF=0.02
		lower_bounds = [fit_gauss["offset"]-0.1, 0, xmin, 0, -5] # q=-5 is T/TF=8

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
		fitted_x = fitted * defaults.od_to_number

		fits.update(
			{
				"peakGauss": fit_gauss["peak"],
				"sigxGauss": fit_gauss["sigx"],
				"sigyGauss": fit_gauss["sigy"],
				"ycGauss": fit_gauss["yc"]
			})
		return (fits, fitted_x, [])

	elif fname == "Fermi 3D":
		(fit_gauss, f_x, f_y) = fitter("Gaussian", data, bounds, xaxis, yaxis, rp)
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

		fitted_x = np.sum(fitted, axis=0) * defaults.od_to_number / height
		fitted_y = np.sum(fitted, axis=1) * defaults.od_to_number / width

		fits.update(
			{
				"peakGauss": fit_gauss["peak"],
				"sigxGauss": fit_gauss["sigx"],
				"sigyGauss": fit_gauss["sigy"]
			})
		return (fits, fitted_x, fitted_y)

