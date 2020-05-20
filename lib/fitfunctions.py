import numpy as np

def gauss_fit(p,r,y):
	(xx,yy) = np.meshgrid(r[0], r[1])
	[offset, peak, xc, yc, sigx, sigy] = p
	return np.ravel(offset + peak*np.exp(-0.5*(xx-xc)**2.0/sigx**2.0 - 0.5*(yy-yc)**2.0/sigy**2.0) - y)

def gaussian(x,y,p):
	[offset, peak, xc, yc, sigx, sigy] = p
	return offset + peak*np.exp(-0.5*(x-xc)**2.0/sigx**2.0 - 0.5*(y-yc)**2.0/sigy**2.0)