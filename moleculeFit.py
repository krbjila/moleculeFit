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

from display import Display, Profiles, Zooms
from loadbar import LoadBar, Autoloader
from roi import ROISelectors
from analysis import AnalysisOptions
from display_options import DisplayOptions
import defaults

import qtreactor.pyqt4reactor
qtreactor.pyqt4reactor.install()
from twisted.internet import reactor

# Temporary
file_path = "K:\\currentmembers\\matsuda\\test_shots_new_sequence\\"
file_number = 0

stack_index = 0

CSatEff = 10000
MAX_OD = 20


class MainWindow(QtGui.QWidget):
	def __init__(self, args):
		super(MainWindow, self).__init__()
		self.setWindowTitle("Molecule fitting")
		self.setFixedSize(defaults.dim_gui[0], defaults.dim_gui[1])

		self.raw = []
		self.region_params = [{}, {}, {}, {}]
		self.populate()

		self.frame = "od"

		# self.plotsomething()

	def populate(self):
		self.layout = QtGui.QHBoxLayout()

		self.profiles = Profiles()
		self.zoom_displays = Zooms()
		self.loadBar = LoadBar()
		self.loadBar.loadSignal.connect(self.load)

		self.display = Display(defaults.dim_display_col[0], defaults.dim_display_col[1])

		self.roi = ROISelectors()
		self.roi.changed.connect(self.roiChanged)
		# self.roi = Region("ROI for big thing", False)
		# self.roi.setValues(defaults.roi_default)
		# self.roi.changed.connect(self.mainROIChanged)

		# self.signal_roi = [
		# 	Region("Signal region " + defaults.state_names[0], False),
		# 	Region("Signal region " + defaults.state_names[1], False)]
		# self.signal_roi[0].setValues(defaults.r00_signal_default)
		# self.signal_roi[0].changed.connect(lambda: self.signalROIChanged(0))

		# self.signal_roi[1].setValues(defaults.r10_signal_default)
		# self.signal_roi[1].changed.connect(lambda: self.signalROIChanged(1))

		# self.background_roi = [
		# 	Region("Background region " + defaults.state_names[0], True),
		# 	Region("Background region " + defaults.state_names[1], True)]
		# self.background_roi[0].setValues(defaults.r00_background_default)
		# self.background_roi[0].changed.connect(lambda: self.backgroundROIChanged(0))

		# self.background_roi[1].setValues(defaults.r10_background_default)
		# self.background_roi[1].changed.connect(lambda: self.backgroundROIChanged(1))

		self.left_layout = QtGui.QVBoxLayout()

		self.left_layout.addWidget(self.loadBar)

		self.left_layout.addWidget(self.roi)

		# self.left_layout.addWidget(self.roi)

		# for w1, w2 in zip(self.signal_roi, self.background_roi):
		# 	self.left_layout.addWidget(w1)
		# 	self.left_layout.addWidget(w2)

		self.analysis = AnalysisOptions()
		
		self.display_options = DisplayOptions()
		self.display_options.changed.connect(self.displayOptionsChanged)

		self.bottom_box = QtGui.QHBoxLayout()
		self.bottom_box.addWidget(self.analysis)
		self.bottom_box.addWidget(self.display_options)

		self.left_layout.addLayout(self.bottom_box)

		self.center_layout = QtGui.QVBoxLayout()
		self.center_layout.addWidget(self.display)

		self.layout.addLayout(self.left_layout)
		self.layout.addLayout(self.center_layout)
		self.layout.addWidget(self.zoom_displays)
		self.layout.addWidget(self.profiles)
		self.setLayout(self.layout)


	def load(self):
		path = str(self.loadBar.pathEdit.text())
		self.currentFilePath = path

		if path:
			try:
				self.raw = np.loadtxt(path, delimiter=',')
				(dy, dx) =  np.shape(self.raw)
				self.raw = np.reshape(self.raw, (defaults.n_frames, dy/defaults.n_frames, dx))

				self.data = {}
				for k,v in defaults.frame_map.items():
					self.data[k] = self.raw[v]

				self.calcOD()

				self.display.setData(self.data)
				# self.display.replot()
				for w in self.profiles.profiles:
					w.setData(self.data)
					# w.replot()
				for w in self.zoom_displays.displays:
					w.setData(self.data)
					# w.replot()
				self.roiChanged()
			except Exception as e:
				print e

	def calcOD(self):
		# Calculate difference and od frames
		s1 = (self.data["shadow"] - self.data["dark"]).astype(np.float64)
		s2 = (self.data["light"] - self.data["dark"]).astype(np.float64)
		od = -np.log(s1 / s2) + (s2 - s1) / CSatEff

		od = np.where(od == np.inf, MAX_OD, od)
		od = np.where(od == -np.inf, -MAX_OD, od)
		od = np.where(od == np.nan, MAX_OD, od) # Check if it makes sense

		self.data["od"] = od

	def displayOptionsChanged(self):
		(frame, vmin, vmax) = self.display_options.getState()

		self.frame = defaults.frame_list[frame]

		self.display.setVlims([vmin, vmax])
		self.display.setFrame(defaults.frame_list[frame])

		for w in self.zoom_displays.displays:
			w.setVlims([vmin, vmax])
			w.setFrame(defaults.frame_list[frame])

		self.roiChanged()

	def roiChanged(self):
		(main, signal, background) = self.roi.getValues()

		self.display.setROI(main)

		for (i, (s, b)) in enumerate(zip(signal, background)):
			self.zoom_displays.displays[2*i].setROI(s)
			self.zoom_displays.displays[2*i + 1].setROI(b)

			self.analyzeROI(s, 2*i)
			self.analyzeROI(b, 2*i+1)

			ps = self.makeRect(2*i, s)
			pb = self.makeRect(2*i+1, b)

			self.display.addPatch(2*i, *ps)
			self.display.addPatch(2*i+1, *pb)

		self.display.replot()
		for (rp,d,p) in zip(self.region_params, self.zoom_displays.displays, self.profiles.profiles):
			rp = rp[self.frame]

			if self.frame == "od":
				d.setIntegratedNumber(rp["sum"]*defaults.od_to_number)

			xline = ((0, defaults.dim_image[0]), (rp["yc"], rp["yc"]))
			yline = ((rp["xc"], rp["xc"]), (0, defaults.dim_image[1]))
			d.setCross(xline, yline)
			d.setOval((rp["xc"], rp["yc"]), 2*rp["sigx"], 2*rp["sigy"])

			xprofile = rp["xprofile"]
			yprofile = rp["yprofile"]
			if self.frame == "od":
				xprofile *= defaults.od_to_number
				yprofile *= defaults.od_to_number
			p.setData((xprofile, yprofile))
			p.setXYRel(rp["xrel"], rp["yrel"])
			d.replot()
			p.replot()

	def getArrayBounds(self, roi):
		xmin = max(roi[0] - roi[2]/2, 0)
		ymin = max(roi[1] - roi[3]/2, 0)

		xmax = min(roi[0] + roi[2]/2, defaults.dim_image[0])
		ymax = min(roi[1] + roi[3]/2, defaults.dim_image[1])
		return (xmin, xmax, ymin, ymax)

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


	def plotsomething(self):

		self.autoloader = Autoloader()
		raw = self.autoloader.load("{}ixon_{}.csv".format(file_path, file_number))
		(dy, dx) =  np.shape(raw)
		raw = np.reshape(raw, (n_frames, dy/n_frames, dx))

		self.data = {}
		for k,v in frame_map.items():
			self.data[k] = raw[v]

		# Calculate difference and od frames
		s1 = (self.data["shadow"] - self.data["dark"]).astype(np.float64)
		s2 = (self.data["light"] - self.data["dark"]).astype(np.float64)
		od = -np.log(s1 / s2) + (s2 - s1) / CSatEff

		od[od == np.inf] = MAX_OD
		od[od == -np.inf] = -MAX_OD
		od[od == np.nan] = 0

		self.data["od"] = od

		self.display.plot(self.data["od"])
		self.profiles.plot([self.data["od"][0]]*4 + [self.data["od"][1]]*4)
		self.zoom_displays.plot([self.data["od"]]*4)


if __name__ == '__main__':
    a = QtGui.QApplication([])
    a.setQuitOnLastWindowClosed(True)
    widget = MainWindow(reactor)

    widget.show()
    reactor.runReturn()
    sys.exit(a.exec_())