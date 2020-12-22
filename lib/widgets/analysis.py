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

import win32com.client

import sys
sys.path.append('../')
import defaults

class AnalysisOptions(QtGui.QWidget):
	changed = pyqtSignal()
	fit = pyqtSignal()
	upload = pyqtSignal()

	def __init__(self):
		super(AnalysisOptions, self).__init__()
		self.setFixedSize(defaults.dim_analysis[0], defaults.dim_analysis[1])
		self.populate()

	# Populate GUI
	def populate(self):
		self.layout = QtGui.QHBoxLayout()

		self.group = QtGui.QGroupBox("Analysis options")
		self.group_layout = QtGui.QGridLayout()

		self.analysisCombo = QtGui.QComboBox()
		for x in defaults.fit_functions:
			self.analysisCombo.addItem(x)
		self.analysisCombo.currentIndexChanged.connect(self.option_changed)

		self.group_layout.addWidget(self.analysisCombo, 0, 0, 1, 2)

		self.fitButton = QtGui.QPushButton("Fit")
		self.fitButton.clicked.connect(self.fit_pressed)
		self.group_layout.addWidget(self.fitButton, 1, 0, 1, 2)

		self.originButton = QtGui.QPushButton("Upload to Origin")
		self.originButton.clicked.connect(self.upload_pressed)
		self.group_layout.addWidget(self.originButton, 2, 0, 1, 2)

		self.autofitLabel = QtGui.QLabel("Auto fit?")
		self.group_layout.addWidget(self.autofitLabel, 3, 0)

		self.autofitCheck = QtGui.QCheckBox()
		self.autofitCheck.stateChanged.connect(self.option_changed)
		self.group_layout.addWidget(self.autofitCheck, 3, 1)

		self.originLabel = QtGui.QLabel("Auto Origin?")
		self.group_layout.addWidget(self.originLabel, 4, 0)

		self.originCheck = QtGui.QCheckBox()
		self.originCheck.stateChanged.connect(self.option_changed)
		self.group_layout.addWidget(self.originCheck, 4, 1)

		self.uploadBothLabel = QtGui.QLabel("Report both fits (G. only)")
		self.group_layout.addWidget(self.uploadBothLabel, 5, 0)

		self.uploadBothCheck = QtGui.QCheckBox()
		self.uploadBothCheck.stateChanged.connect(self.option_changed)
		self.group_layout.addWidget(self.uploadBothCheck, 5, 1)

		self.group_layout.setColumnStretch(2,1)
		self.group.setLayout(self.group_layout)
		self.group.setStyleSheet(defaults.style_sheet)

		self.layout.addWidget(self.group)
		self.setLayout(self.layout)

	def fit_pressed(self):
		self.fit.emit()

	def upload_pressed(self):
		self.upload.emit()

	def getValues(self):
		out = (
			defaults.fit_functions[self.analysisCombo.currentIndex()],
			self.autofitCheck.isChecked(),
			self.originCheck.isChecked(),
			self.uploadBothCheck.isChecked()
		)
		return out

	def option_changed(self):
		self.changed.emit()

	def upload_origin(self, data, fitfunction=None, uploadBoth=False):
		pid = 'Origin.ApplicationSI'
		origin = win32com.client.Dispatch(pid)

		if "Gaussian" in fitfunction:
			if not uploadBoth:
				short_name = "KRbFKGauss1"
				long_name = "KRb FK Gaussian"

				if origin.FindWorksheet(short_name) is None:
					origin.CreatePage(2, short_name, "KRbFKGauss")
			else:
				short_name = "KRbBothG1"
				long_name = "KRb Both Gauss"

				if origin.FindWorksheet(short_name) is None:
					origin.CreatePage(2, short_name, "KRbBothG")

		elif fitfunction == "Fermi 2D":
			short_name = "KRbFermi2D1"
			long_name = "KRb FK Fermi-Dirac 2D"

			if origin.FindWorksheet(short_name) is None:
				origin.CreatePage(2, short_name, "KRbFermi2D")
		elif fitfunction == "Fermi 3D":
			short_name = "KRbFermi3D1"
			long_name = "KRb FK Fermi-Dirac 3D"

			if origin.FindWorksheet(short_name) is None:
				origin.CreatePage(2, short_name, "KRbFermi3D")
		else:
			short_name = "KRbInt1"
			long_name = "KRb Integrated"

			if origin.FindWorksheet(short_name) is None:
				origin.CreatePage(2, short_name, "KRbInt")
		origin.Execute("{}!page.longname$ = {}".format(short_name, long_name))

		for (i, x) in enumerate(data):
			ret = origin.PutWorksheet("[{}]Sheet1".format(short_name), x, -1, i)