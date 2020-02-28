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

class ROISelectors(QtGui.QWidget):
	subregion_changed = pyqtSignal()
	main_changed = pyqtSignal()
	currentValues = ()

	def __init__(self):
		super(ROISelectors, self).__init__()
		self.populate()
		self.connect()

	def populate(self):
		self.layout = QtGui.QVBoxLayout()

		self.roi = Region("ROI for big thing", False)
		self.roi.setValues(defaults.roi_default)
		self.roi.setStyleSheet(defaults.style_sheet)

		self.signal_roi = [
			Region("Signal region " + defaults.state_names[0], False),
			Region("Signal region " + defaults.state_names[1], False)]
		self.signal_roi[0].setValues(defaults.r00_signal_default)
		self.signal_roi[1].setValues(defaults.r10_signal_default)

		self.background_roi = [
			Region("Background region " + defaults.state_names[0], True),
			Region("Background region " + defaults.state_names[1], True)]
		self.background_roi[0].setValues(defaults.r00_background_default)
		self.background_roi[1].setValues(defaults.r10_background_default)

		self.layout.addWidget(self.roi)
		for (i, (w1, w2)) in enumerate(zip(self.signal_roi, self.background_roi)):
			groupbox = QtGui.QGroupBox("Region for " + defaults.state_names[i])
			gl = QtGui.QVBoxLayout()

			gl.addWidget(w1)
			gl.addWidget(w2)
			groupbox.setLayout(gl)
			groupbox.setStyleSheet(defaults.style_sheet)

			self.layout.addWidget(groupbox)

		self.setLayout(self.layout)

	def getValues(self):
		self._getValues()
		return self.currentValues

	def connect(self):
		self.roi.changed.connect(self.mainROIChanged)
		for w in self.signal_roi + self.background_roi:
			w.changed.connect(self.subROIChanged)
		for w in self.background_roi:
			w.checked.connect(self.checked)

	def checked(self):
		self.handleDefaults()
		self.subROIChanged()

	def handleDefaults(self):
		for (w1, w2) in zip(self.signal_roi, self.background_roi):
			if w2.default_checked:
				roi = w1.getValues()
				roi[1] += roi[3]
				w2.setValues(roi)

	def _getValues(self):
		main_roi = self.roi.getValues()

		signal_roi = []
		for w in self.signal_roi:
			signal_roi.append(w.getValues())

		self.handleDefaults()
		for (ws,wb) in zip(self.signal_roi, self.background_roi):
			rs = ws.getValues()
			wb.setSizes(rs[2], rs[3])

		background_roi = []
		for w in self.background_roi:
			background_roi.append(w.getValues())

		self.currentValues = (main_roi, signal_roi, background_roi)

	def mainROIChanged(self):
		self._getValues()
		self.main_changed.emit()

	def subROIChanged(self):
		self._getValues()
		self.subregion_changed.emit()


class Region(QtGui.QWidget):
	changed = pyqtSignal()
	checked = pyqtSignal()

	def __init__(self, title="", default_button=False):
		super(Region, self).__init__()
		self.setFixedSize(defaults.dim_roi[0], defaults.dim_roi[1])
		self.title = title
		self.default = default_button
		self.populate()

		self.default_checked = False

	# Populate GUI
	def populate(self):
		self.layout = QtGui.QHBoxLayout()

		self.edits = []
		self.group = QtGui.QGroupBox(self.title)
		gl1 = QtGui.QHBoxLayout()
		gl2 = QtGui.QHBoxLayout()
		self.group_layout = QtGui.QVBoxLayout()

		if self.default:
			self.defaultLabel = QtGui.QLabel("Default?")
			self.defaultCheck = QtGui.QCheckBox()
			self.defaultCheck.stateChanged.connect(self.check)

			gl1.addWidget(self.defaultLabel)
			gl1.addWidget(self.defaultCheck)
			gl1.addStretch(10)


		self.xcLabel = QtGui.QLabel("Xc")
		gl2.addWidget(self.xcLabel)

		self.xcEdit = QtGui.QLineEdit()
		gl2.addWidget(self.xcEdit)
		self.edits.append(self.xcEdit)

		self.ycLabel = QtGui.QLabel("Yc")
		gl2.addWidget(self.ycLabel)

		self.ycEdit = QtGui.QLineEdit()
		gl2.addWidget(self.ycEdit)
		self.edits.append(self.ycEdit)

		self.dxLabel = QtGui.QLabel("dx")
		gl2.addWidget(self.dxLabel)

		self.dxEdit = QtGui.QLineEdit()
		gl2.addWidget(self.dxEdit)
		self.edits.append(self.dxEdit)
		if self.default:
			self.dxEdit.setDisabled(True)

		self.dyLabel = QtGui.QLabel("dy")
		gl2.addWidget(self.dyLabel)

		self.dyEdit = QtGui.QLineEdit()
		gl2.addWidget(self.dyEdit)
		self.edits.append(self.dyEdit)
		if self.default:
			self.dyEdit.setDisabled(True)

		for widget in self.edits:
			widget.returnPressed.connect(self.edited)

		self.group_layout.addLayout(gl1)
		self.group_layout.addLayout(gl2)

		self.group.setLayout(self.group_layout)

		self.layout.addWidget(self.group)
		self.setLayout(self.layout)


	def setValues(self, vals):
		for w, v in zip(self.edits, vals):
			w.setText(str(int(float(v))))

	def getValues(self):
		out = []
		for widget in self.edits:
			out.append(int(widget.text()))
		return out

	def check(self):
		if self.defaultCheck.isChecked():
			self.default_checked = True
			for widget in self.edits:
				widget.setDisabled(True)
			self.checked.emit()
		else:
			self.default_checked = False
			for widget in self.edits:
				widget.setDisabled(False)

	def edited(self):
		try:
			for widget in self.edits:
				widget.setText(str(int(float(widget.text()))))
			self.changed.emit()
		# Just don't 
		except:
			pass

	def setSizes(self, dx, dy):
		self.dxEdit.setText(str(int(dx)))
		self.dyEdit.setText(str(int(dy)))
