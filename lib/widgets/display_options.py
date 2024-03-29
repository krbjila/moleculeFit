from PyQt4 import QtGui, QtCore, Qt
from PyQt4.QtCore import pyqtSignal
from twisted.internet.defer import inlineCallbacks
import twisted.internet.error

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

from matplotlib import pyplot as plt

import time

import numpy as np
from copy import deepcopy

import sys
sys.path.append('../')
import defaults

class DisplayOptions(QtGui.QWidget):
	changed = pyqtSignal()

	odLimits = (-10, 10)
	odStep = 0.1
	countLimits = (-2e4, 2e4)
	countStep = 100

	def __init__(self):
		super(DisplayOptions, self).__init__()
		self.setFixedSize(defaults.dim_display_opt[0], defaults.dim_display_opt[0])
		self.populate()

		self.od = [0, 0.5]
		self.counts = [500, 5000]

	def populate(self):
		self.layout = QtGui.QGridLayout()

		self.group = QtGui.QGroupBox("Display options")
		self.group_layout = QtGui.QGridLayout()

		self.frameLabel = QtGui.QLabel("Frame")

		self.frameCombo = QtGui.QComboBox()
		for x in defaults.frame_list:
			self.frameCombo.addItem(x)
		self.frameCombo.currentIndexChanged.connect(self.indexChanged)

		self.vmaxLabel = QtGui.QLabel("Max")
		self.vmaxEdit = QtGui.QLineEdit("1.0")
		self.vmaxEdit.editingFinished.connect(self.edited)

		self.vminLabel = QtGui.QLabel("Min")
		self.vminEdit = QtGui.QLineEdit("0.0")
		self.vminEdit.editingFinished.connect(self.edited)


		crosshairs = QtGui.QGroupBox("Crosshairs")
		gl = QtGui.QGridLayout()
		self.xLabel = QtGui.QLabel("x")
		self.xEdit = QtGui.QLineEdit("0.0")
		self.xEdit.editingFinished.connect(self.edited)

		self.yLabel = QtGui.QLabel("y")
		self.yEdit = QtGui.QLineEdit("0.0")
		self.yEdit.editingFinished.connect(self.edited)

		gl.addWidget(self.xLabel, 0, 0)
		gl.addWidget(self.xEdit, 0, 1)
		gl.addWidget(self.yLabel, 1, 0)
		gl.addWidget(self.yEdit, 1, 1)
		crosshairs.setLayout(gl)

		self.group_layout.addWidget(self.frameLabel, 0, 0, 1, 1)
		self.group_layout.addWidget(self.frameCombo, 0, 1, 1, 1)

		self.group_layout.addWidget(self.vminLabel, 1, 0, 1, 1)
		self.group_layout.addWidget(self.vminEdit, 1, 1, 1, 1)

		self.group_layout.addWidget(self.vmaxLabel, 2, 0, 1, 1)
		self.group_layout.addWidget(self.vmaxEdit, 2, 1, 1, 1)

		self.group_layout.addWidget(crosshairs, 3, 0, 2, 2)

		self.group_layout.setColumnStretch(2, 1)

		self.group.setLayout(self.group_layout)
		self.group.setStyleSheet(defaults.style_sheet)

		self.layout.addWidget(self.group)
		self.setLayout(self.layout)

	def indexChanged(self):
		if self.frameCombo.currentIndex():
			vals = self.counts
		else:
			vals = self.od

		self.setVlims(vals[0], vals[1])
		self.edited()

	def setVlims(self, vmin, vmax):
		minmax = self.minmax(vmin, vmax)

		if self.frameCombo.currentIndex():
			self.vminEdit.setText(str(int(minmax[0])))
			self.vmaxEdit.setText(str(int(minmax[1])))
		else:
			self.vminEdit.setText("{:.2f}".format(minmax[0]))
			self.vmaxEdit.setText("{:.2f}".format(minmax[1]))

	def setCrosshairs(self, x, y):
		self.xEdit.setText("{:.1f}".format(x))
		self.yEdit.setText("{:.1f}".format(y))

	def edited(self):
		try:
			vmin = float(self.vminEdit.text())
			vmax = float(self.vmaxEdit.text())

			minmax = self.minmax(vmin, vmax)
			self.setVlims(vmin, vmax)

			if self.frameCombo.currentIndex():
				self.counts = [minmax[0], minmax[1]]
			else:
				self.od = [minmax[0], minmax[1]]

			x = float(self.xEdit.text())
			y = float(self.yEdit.text())
			self.setCrosshairs(x, y)

			self.changed.emit()
		except Exception as e:
			print "Exception in DisplayOptions.edited: "

	def minmax(self, v1, v2):
		temp_max = max(v1, v2)
		temp_min = min(v1, v2)
		return (temp_min, temp_max)

	def getState(self):
		return (self.frameCombo.currentIndex(), float(self.vminEdit.text()), float(self.vmaxEdit.text()), float(self.xEdit.text()), float(self.yEdit.text()))