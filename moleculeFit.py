from PyQt4 import QtGui, QtCore, Qt
from PyQt4.QtCore import pyqtSignal
from twisted.internet.defer import inlineCallbacks
import twisted.internet.error

from matplotlib.patches import Rectangle

import time

import numpy as np
from copy import deepcopy

import sys
sys.path.append("./lib/")
sys.path.append("./lib/widgets/")
sys.path.append("./lib/qt4reactor/")

from display import PlotGroup
from loadbar import LoadBar, Autoloader
from roi import ROISelectors
from analysis import AnalysisOptions
from display_options import DisplayOptions
import defaults

import qtreactor.pyqt4reactor
qtreactor.pyqt4reactor.install()
from twisted.internet import reactor


class MainWindow(QtGui.QWidget):
	def __init__(self, args):
		super(MainWindow, self).__init__()
		self.setWindowTitle("Molecule fitting")
		self.setFixedSize(defaults.dim_gui[0], defaults.dim_gui[1])

		self.raw = []
		self.region_params = [{}, {}, {}, {}]
		self.initialize()
		self.populate()

		self.frame = "od"

	def initialize(self):

		self.loadBar = LoadBar()
		self.loadBar.loadSignal.connect(self.load)

		self.roi = ROISelectors()
		self.roi.changed.connect(self.roiChanged)

		self.analysis = AnalysisOptions()
		
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
		path = str(self.loadBar.pathEdit.text())
		self.currentFilePath = path

		if path:
			self.fileName = path.split('/')[-1]
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

	def displayOptionsChanged(self):
		(frame, vmin, vmax, x, y) = self.display_options.getState()

		self.frame = defaults.frame_list[frame]

		self.plotGroup.setVlims([vmin, vmax])
		self.plotGroup.setFrame("m", 0, self.frame)
		for i in range(self.plotGroup.grid_size):
			self.plotGroup.setFrame("d", i, self.frame)

		self.plotGroup.setCross("m", 0, x, y)
		self.roiChanged()

	def roiChanged(self):
		(main, signal, background) = self.roi.getValues()

		self.plotGroup.setROI("m", 0, main)
		for (i, (s, b)) in enumerate(zip(signal, background)):
			self.plotGroup.setROI("d", 2*i, s)
			self.plotGroup.setROI("d", 2*i+1, b)

			self.analyzeROI(s, 2*i)
			self.analyzeROI(b, 2*i+1)

			ps = self.makeRect(2*i, s) + (defaults.rect_colors[2*i],)
			pb = self.makeRect(2*i+1, b) + (defaults.rect_colors[2*i+1],)

			self.plotGroup.setRectMain(2*i, *ps)
			self.plotGroup.setRectMain(2*i+1, *pb)

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

		self.analysis.upload_origin(self.paramsToOrigin())
		self.plotGroup.replot()

	def getArrayBounds(self, roi):
		xmin = max(roi[0] - roi[2]/2, 0)
		ymin = max(roi[1] - roi[3]/2, 0)

		xmax = min(roi[0] + roi[2]/2, defaults.dim_image[0])
		ymax = min(roi[1] + roi[3]/2, defaults.dim_image[1])
		return (xmin, xmax, ymin, ymax)

	def paramsToOrigin(self):
		arr = [
			self.fileName,
			self.region_params[0]["od"]["sum"]*defaults.od_to_number, 		# N 0
			self.region_params[1]["od"]["sum"]*defaults.od_to_number, 		# Background 0
			self.region_params[0]["od"]["sigx"]*defaults.pixel_size*1e6, 	# sigx0 (um)
			self.region_params[0]["od"]["sigy"]*defaults.pixel_size*1e6, 	# sigy0 (um)
			self.region_params[0]["od"]["xc"], 								# xc0 (px)
			self.region_params[0]["od"]["yc"], 								# yc0 (px)
			self.region_params[2]["od"]["sum"]*defaults.od_to_number, 		# N 1
			self.region_params[3]["od"]["sum"]*defaults.od_to_number, 		# Background 1
			self.region_params[2]["od"]["sigx"]*defaults.pixel_size*1e6, 	# sigx1 (um)
			self.region_params[2]["od"]["sigy"]*defaults.pixel_size*1e6, 	# sigy1 (um)
			self.region_params[2]["od"]["xc"], 								# xc1 (px)
			self.region_params[2]["od"]["yc"], 								# yc1 (px)
		]
		return arr


	def analyzeROI(self, roi, index):
		(xmin, xmax, ymin, ymax) = self.getArrayBounds(roi)
		frame = self.frame

		data = self.data[frame][ymin:ymax, xmin:xmax]
		xaxis = np.arange(xmin, xmax, 1)
		yaxis = np.arange(ymin, ymax, 1)

		integrated = np.sum(data)
		xprofile = np.sum(data, axis=0)
		yprofile = np.sum(data, axis=1)

		xcom = np.sum(xprofile*xaxis)/integrated
		ycom = np.sum(yprofile*yaxis)/integrated

		try:
			xsig = np.sqrt(np.sum(xprofile*xaxis**2)/integrated - xcom**2)
			ysig = np.sqrt(np.sum(yprofile*yaxis**2)/integrated - ycom**2)
		except:
			xsig = 0
			ysig = 0

		d = {
			"sum": integrated,
			"xprofile": xprofile,
			"yprofile": yprofile,
			"xc": xcom,
			"xrel": roi[0] - xcom,
			"yc": ycom,
			"yrel": roi[1] - ycom,
			"sigx": xsig,
			"sigy": ysig
		}

		self.region_params[index].update({frame: d})

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