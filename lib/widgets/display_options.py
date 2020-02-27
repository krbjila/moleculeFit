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

		self.od = [0, 1.0]
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
		# self.vmaxEdit = QtGui.QDoubleSpinBox()
		# self.vmaxEdit.valueChanged.connect(self.edited)
		# self.vmaxEdit.setSingleStep(self.odStep)
		# self.vmaxEdit.setRange(*self.odLimits)
		# self.vmaxEdit.setValue(1.0)

		self.vminLabel = QtGui.QLabel("Min")
		self.vminEdit = QtGui.QLineEdit("0.0")
		self.vminEdit.editingFinished.connect(self.edited)
		# self.vminEdit = QtGui.QDoubleSpinBox()
		# self.vminEdit.valueChanged.connect(self.edited)
		# self.vminEdit.setSingleStep(self.odStep)
		# self.vminEdit.setRange(*self.odLimits)
		# self.vminEdit.setValue(0)

		self.group_layout.addWidget(self.frameLabel, 0, 0, 1, 1)
		self.group_layout.addWidget(self.frameCombo, 0, 1, 1, 1)

		self.group_layout.addWidget(self.vminLabel, 1, 0, 1, 1)
		self.group_layout.addWidget(self.vminEdit, 1, 1, 1, 1)

		self.group_layout.addWidget(self.vmaxLabel, 2, 0, 1, 1)
		self.group_layout.addWidget(self.vmaxEdit, 2, 1, 1, 1)

		self.group_layout.setColumnStretch(2, 1)

		self.group.setLayout(self.group_layout)

		self.layout.addWidget(self.group)
		self.setLayout(self.layout)

	def indexChanged(self):
		if self.frameCombo.currentIndex():
			vals = self.counts
		else:
			vals = self.od

		self.setValues(vals[0], vals[1])
		self.edited()

	def setValues(self, vmin, vmax):
		minmax = self.minmax(vmin, vmax)

		if self.frameCombo.currentIndex():
			self.vminEdit.setText(str(int(minmax[0])))
			self.vmaxEdit.setText(str(int(minmax[1])))
		else:
			self.vminEdit.setText("{:.2f}".format(minmax[0]))
			self.vmaxEdit.setText("{:.2f}".format(minmax[1]))

	def edited(self):
		try:
			vmin = float(self.vminEdit.text())
			vmax = float(self.vmaxEdit.text())

			minmax = self.minmax(vmin, vmax)
			self.setValues(vmin, vmax)

			if self.frameCombo.currentIndex():
				self.counts = [minmax[0], minmax[1]]
			else:
				self.od = [minmax[0], minmax[1]]

			self.changed.emit()
		# Just don't 
		except:
			pass

	def minmax(self, v1, v2):
		temp_max = max(v1, v2)
		temp_min = min(v1, v2)
		return (temp_min, temp_max)

	def getState(self):
		return (self.frameCombo.currentIndex(), float(self.vminEdit.text()), float(self.vmaxEdit.text()))