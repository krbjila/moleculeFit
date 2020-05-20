from PyQt4 import QtGui, QtCore, Qt
from PyQt4.QtCore import pyqtSignal
from twisted.internet.defer import inlineCallbacks
import twisted.internet.error

from matplotlib.patches import Rectangle

import time

import numpy as np
from scipy.optimize import least_squares
from copy import deepcopy

import sys
sys.path.append("./lib/")
sys.path.append("./lib/widgets/")
sys.path.append("./lib/qt4reactor/")

from display import PlotGroup
from loadbar import LoadBar
from roi import ROISelectors
from analysis import AnalysisOptions
from display_options import DisplayOptions
import defaults
import fitfunctions

import os

import qtreactor.pyqt4reactor
qtreactor.pyqt4reactor.install()
from twisted.internet import reactor


class MainWindow(QtGui.QWidget):
	def __init__(self, reactor):
		super(MainWindow, self).__init__()

		self.reactor = reactor

		self.setWindowTitle("Molecule fitting")
		self.setFixedSize(defaults.dim_gui[0], defaults.dim_gui[1])

		self.raw = []
		self.region_params = [{}, {}, {}, {}]
		self.fits = [{}, {}]
		self.fit_function = ""
		self.auto_fit = False
		self.auto_origin = False

		self.autoload_deferred = None

		self.initialize()
		self.populate()

		self.frame = "od"

	def initialize(self):

		self.loadBar = LoadBar()
		self.loadBar.loadSignal.connect(self.load)
		self.loadBar.autoloadSignal.connect(self.autoload)

		self.roi = ROISelectors()
		self.roi.main_changed.connect(self.mainROIChanged)
		self.roi.subregion_changed.connect(self.subROIChanged)

		self.analysis = AnalysisOptions()
		self.analysis.changed.connect(self.analysisOptionsChanged)
		self.analysis.fit.connect(self.fit)
		self.analysis.upload.connect(self.upload)
		
		self.display_options = DisplayOptions()
		self.display_options.changed.connect(self.displayOptionsChanged)

		self.plotGroup = PlotGroup()
		self.plotGroup.crosshairs_placed.connect(self.crosshairsChanged)
		self.plotGroup.setFixedSize(defaults.dim_plot_group[0], defaults.dim_plot_group[1])

	def populate(self):
		self.layout = QtGui.QHBoxLayout()	
		self.left_layout = QtGui.QVBoxLayout()

		self.left_layout.addWidget(self.loadBar)
		self.left_layout.addWidget(self.roi)

		self.bottom_box = QtGui.QHBoxLayout()
		self.bottom_box.addWidget(self.analysis)
		self.bottom_box.addWidget(self.display_options)

		self.left_layout.addLayout(self.bottom_box)

		self.layout.addLayout(self.left_layout)
		self.layout.addWidget(self.plotGroup)

		self.setLayout(self.layout)


	def load(self):
		path = str(self.loadBar.getPath())
		self.currentFilePath = path

		if path:
			self.fileName = path.split('\\')[-1]
			self.fileNumber = int(self.fileName.split('_')[-1].split('.csv')[0])

			self.raw = np.loadtxt(path, delimiter=',')
			(dy, dx) =  np.shape(self.raw)
			self.raw = np.reshape(self.raw, (defaults.n_frames, dy/defaults.n_frames, dx))

			self.data = {}
			for k,v in defaults.frame_map.items():
				self.data[k] = self.raw[v]

			self.calcOD()

			self.plotGroup.setData("m", 0, self.data)
			for i in range(self.plotGroup.grid_size):
				self.plotGroup.setData("d", i, self.data)

			self.roiChanged()

	def autoload(self, autoload):
		if autoload:
			filenumber = self.loadBar.getFileNumber()
			filename = defaults.filename.format(filenumber)
			path = str(self.loadBar.getPath()).split(defaults.filebase)[0] + filename

			self._autoload(filenumber, path)
		else:
			self.autoload_deferred.cancel()

	def _autoload(self, filenumber, path):
		if os.path.exists(path):
			# File exists, so increment everything
			next_number = filenumber + 1
			next_file = defaults.filename.format(next_number)
			next_path = path.split(defaults.filebase)[0] + next_file

			self.loadBar.setFilePath(path)
			self.loadBar.setFileNumber(next_number)
			self.reactor.callLater(defaults.wait_for_load, self.load)
			self.autoload_deferred = self.reactor.callLater(defaults.autoload_loop, self._autoload, next_number, next_path)
		else:
			self.autoload_deferred = self.reactor.callLater(defaults.autoload_loop, self._autoload, filenumber, path)

	def fit(self):
		self._fit()
		self.plotGroup.replot()

	def _fit(self):
		(main, signal, background) = self.roi.getValues()
		(fitfunction, auto_fit, auto_origin) = self.analysis.getValues()

		for (i, roi) in enumerate(signal):
			(xmin, xmax, ymin, ymax) = self.getArrayBounds(roi)
			data = self.data["od"][ymin:ymax, xmin:xmax]
			xaxis = np.arange(xmin, xmax, 1)
			yaxis = np.arange(ymin, ymax, 1)

			rp = self.region_params[2*i]["od"]

			if fitfunction == "Gaussian":
				guess = [0, 0.5, rp["xc"], rp["yc"], rp["sigx"], rp["sigy"]]
				upper_bounds = [defaults.max_od, defaults.max_od, xmax, ymax, xmax-xmin, ymax-ymin]
				lower_bounds = [-defaults.max_od, 0, xmin, ymin, 0, 0]

				res = least_squares(fitfunctions.gauss_fit, guess, args=([xaxis, yaxis], data), bounds=(lower_bounds, upper_bounds))
				if not res.success:
					print "Warning: fit did not converge."

				self.fits[i] = {
					"f": "Gaussian",
					"offset": res.x[0],
					"peak": res.x[1],
					"xc": res.x[2],
					"yc": res.x[3],
					"sigx": res.x[4],
					"sigy": res.x[5]
				}

				(width, height) = (xmax - xmin, ymax - ymin)
				fitted = fitfunctions.gauss_fit(res.x, [xaxis, yaxis], 0)
				fitted = np.reshape(fitted, (height, width))

				fitted_x = np.sum(fitted, axis=0) * defaults.od_to_number / height
				fitted_y = np.sum(fitted, axis=1) * defaults.od_to_number / width

			self.plotGroup.setFitData("p", 2*i, (fitted_x, fitted_y))
			self.plotGroup.setFitAxes("p", 2*i, (xaxis - int(rp["xc"]), yaxis - int(rp["yc"])))

		if auto_origin:
			self.upload()

	def upload(self):
		if self.region_params[0]:
			if self.fits[0]:
				f = self.fits[0]['f']
				self.analysis.upload_origin(self.fitParamsToOrigin(f), f)
			else:
				self.analysis.upload_origin(self.integrationParamsToOrigin())

	def calcOD(self):
		# Calculate difference and od frames
		s1 = (self.data["shadow"] - self.data["dark"]).astype(np.float64)
		s2 = (self.data["light"] - self.data["dark"]).astype(np.float64)
		od = -np.log(s1 / s2) + (s2 - s1) / defaults.c_sat_eff

		od = np.where(od == np.inf, defaults.max_od, od)
		od = np.where(od == -np.inf, -defaults.max_od, od)
		od = np.where(od == np.nan, defaults.max_od, od) # Check if it makes sense

		self.data["od"] = od

	def crosshairsChanged(self, x, y):
		self.display_options.setCrosshairs(x, y)

	def analysisOptionsChanged(self):
		(fit_function, autofit, origin) = self.analysis.getValues()
		self.fit_function = fit_function
		self.auto_fit = autofit
		self.auto_origin = origin

	def displayOptionsChanged(self):
		(frame, vmin, vmax, x, y) = self.display_options.getState()

		self.frame = defaults.frame_list[frame]

		self.plotGroup.setVlims([vmin, vmax])
		self.plotGroup.setFrame("m", 0, self.frame)
		for i in range(self.plotGroup.grid_size):
			self.plotGroup.setFrame("d", i, self.frame)

		self.plotGroup.setCross("m", 0, x, y)
		self.mainROIChanged()

	def roiChanged(self):
		self._mainROIChanged()
		self._subROIChanged()
		self.plotGroup.replot()

	def _mainROIChanged(self):
		(main, signal, background) = self.roi.getValues()
		self.plotGroup.setROI("m", 0, main)
		for (i, (s, b)) in enumerate(zip(signal, background)):
			self.plotGroup.setROI("d", 2*i, s)
			self.plotGroup.setROI("d", 2*i+1, b)

			self.analyzeROI(s, b, 2*i)

			ps = self.makeRect(2*i, s) + (defaults.rect_colors[2*i],)
			pb = self.makeRect(2*i+1, b) + (defaults.rect_colors[2*i+1],)

			self.plotGroup.setRectMain(2*i, *ps)
			self.plotGroup.setRectMain(2*i+1, *pb)

	def mainROIChanged(self):
		self._mainROIChanged()
		self.plotGroup.replot()

	def _subROIChanged(self):
		self._mainROIChanged()
		for (i, rp) in enumerate(self.region_params):
			rp = rp[self.frame]

			if self.frame == "od":
				self.plotGroup.setNumber("d", i, rp["sum"]*defaults.od_to_number)

			self.plotGroup.setCross("d", i, rp["xc"], rp["yc"])
			self.plotGroup.setOvalDisplay(i, (rp["xc"], rp["yc"]), 2*rp["sigx"], 2*rp["sigy"], defaults.rect_colors[i])

			xprofile = rp["xprofile"]
			yprofile = rp["yprofile"]
			if self.frame == "od":
				xprofile *= defaults.od_to_number
				yprofile *= defaults.od_to_number
			self.plotGroup.setData("p", i, (xprofile, yprofile))
			self.plotGroup.setXYRelProfile(i, (rp["xrel"], rp["yrel"]))

		if self.auto_fit:
			self._fit()
		else:
			self.fits = [{}, {}]
			if self.auto_origin:
				self.upload()

	def subROIChanged(self):
		self._subROIChanged()
		self.plotGroup.replot()

	def getArrayBounds(self, roi):
		xmin = max(roi[0] - roi[2]/2, 0)
		ymin = max(roi[1] - roi[3]/2, 0)

		xmax = min(roi[0] + roi[2]/2, defaults.dim_image[0])
		ymax = min(roi[1] + roi[3]/2, defaults.dim_image[1])
		return (xmin, xmax, ymin, ymax)

	def integrationParamsToOrigin(self):
		(main, signal, background) = self.roi.getValues()

		arr = [
			self.fileName,
			self.region_params[0]["od"]["sum"]*defaults.od_to_number, 		# N 0
			self.region_params[1]["od"]["sum"]*defaults.od_to_number, 		# Background 0
			signal[0][2],													# Integration region 0 width
			signal[0][3],													# Integration region 0 height
			self.region_params[0]["od"]["sigx"]*defaults.pixel_size*1e6, 	# sigx0 (um)
			self.region_params[0]["od"]["sigy"]*defaults.pixel_size*1e6, 	# sigy0 (um)
			self.region_params[0]["od"]["xc"], 								# xc0 (px)
			self.region_params[0]["od"]["yc"], 								# yc0 (px)
			self.region_params[2]["od"]["sum"]*defaults.od_to_number, 		# N 1
			self.region_params[3]["od"]["sum"]*defaults.od_to_number, 		# Background 1
			signal[1][2],													# Integration region 1 width
			signal[1][3],													# Integration region 1 height
			self.region_params[2]["od"]["sigx"]*defaults.pixel_size*1e6, 	# sigx1 (um)
			self.region_params[2]["od"]["sigy"]*defaults.pixel_size*1e6, 	# sigy1 (um)
			self.region_params[2]["od"]["xc"], 								# xc1 (px)
			self.region_params[2]["od"]["yc"], 								# yc1 (px)
		]
		return arr

	def fitParamsToOrigin(self, fitfunction):
		(main, signal, background) = self.roi.getValues()

		if fitfunction == "Gaussian":		
			arr = [
				self.fileName,
				signal[0][2],										# Fit region 0 width
				signal[0][3],										# Fit region 0 height
				self.fits[0]["peak"],								# Peak OD 0
				self.fits[0]["sigx"]*defaults.pixel_size*1e6, 		# sigx0 (um)
				self.fits[0]["sigy"]*defaults.pixel_size*1e6, 		# sigx0 (um)
				self.fits[0]["xc"], 								# xc0 (px)
				self.fits[0]["yc"], 								# yc0 (px)
				self.fits[0]["offset"],								# Offset 0
				# signal[1][2],										# Fit region 1 width
				# signal[1][3],										# Fit region 1 height
				# self.fits[1]["peak"],								# Peak OD 1
				# self.fits[1]["sigx"]*defaults.pixel_size*1e6, 		# sigx1 (um)
				# self.fits[1]["sigy"]*defaults.pixel_size*1e6, 		# sigx1 (um)
				# self.fits[1]["xc"], 								# xc1 (px)
				# self.fits[1]["yc"], 								# yc1 (px)
				# self.fits[1]["offset"],								# Offset 1
			]
			return arr


	def analyzeROI(self, s_roi, b_roi, index):
		frame = self.frame

		d = {}
		for (sb, roi) in zip(['s', 'b'], [s_roi, b_roi]):
			x = {}
			(xmin, xmax, ymin, ymax) = self.getArrayBounds(roi)
			x['xmin'] = xmin
			x['xmax'] = xmax
			x['ymin'] = ymin
			x['ymax'] = ymax

			x['data'] = self.data[frame][ymin:ymax, xmin:xmax]
			x['xaxis'] = np.arange(xmin, xmax, 1)
			x['yaxis'] = np.arange(ymin, ymax, 1)
			d[sb] = x

		for (sb, roi) in zip(['s', 'b'], [s_roi, b_roi]):
			if sb == 's':
				shift = np.array(d['s']['data']) - np.array(d['b']['data'])
			else:
				shift = np.array(d[sb]['data'])

			integrated = np.sum(d[sb]['data'])
			xp = np.sum(d[sb]['data'], axis=0)/roi[3]
			yp = np.sum(d[sb]['data'], axis=1)/roi[2]

			shift_int = np.sum(shift)
			shift_xp = np.sum(shift, axis=0)
			shift_yp = np.sum(shift, axis=1)

			xcom = np.sum(shift_xp*d[sb]['xaxis'])/shift_int
			ycom = np.sum(shift_yp*d[sb]['yaxis'])/shift_int

			# Clip COM to ROI
			xcom = max(min(xcom, d[sb]['xmax']), d[sb]['xmin'])
			ycom = max(min(ycom, d[sb]['ymax']), d[sb]['ymin'])

			xvar = np.sum(shift_xp*(d[sb]['xaxis'] - xcom)**2)/shift_int
			yvar = np.sum(shift_yp*(d[sb]['yaxis'] - ycom)**2)/shift_int

			if xvar > 0:
				xsig = np.sqrt(xvar)
			else:
				xsig = 0
			if yvar > 0:
				ysig = np.sqrt(yvar)
			else:
				ysig = 0

			dd = {
				"roi": roi,
				"sum": integrated,
				"xprofile": xp,
				"yprofile": yp,
				"xc": xcom,
				"xrel": roi[0] - xcom,
				"yc": ycom,
				"yrel": roi[1] - ycom,
				"sigx": xsig,
				"sigy": ysig
			}
			if sb == 's':
				self.region_params[index].update({frame: dd})
			else:
				self.region_params[index + 1].update({frame: dd})

	def makeRect(self, index, roi):
		x = max(roi[0] - roi[2]/2, 0)
		y = max(roi[1] - roi[3]/2, 0)
		return ((x,y), roi[2], roi[3])


if __name__ == '__main__':
    a = QtGui.QApplication([])
    a.setQuitOnLastWindowClosed(True)
    widget = MainWindow(reactor)

    widget.show()
    reactor.runReturn()
    sys.exit(a.exec_())