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
		self.analysisCombo.addItem("Integrate")
		self.analysisCombo.addItem("Gaussian fit")
		self.group_layout.addWidget(self.analysisCombo, 0, 0, 1, 2)

		self.fitButton = QtGui.QPushButton("Fit?")
		self.group_layout.addWidget(self.fitButton, 1, 0, 1, 2)

		self.originButton = QtGui.QPushButton("Upload to Origin?")
		self.group_layout.addWidget(self.originButton, 2, 0, 1, 2)

		self.autofitLabel = QtGui.QLabel("Auto fit?")
		self.group_layout.addWidget(self.autofitLabel, 3, 0)

		self.autofitCheck = QtGui.QCheckBox()
		self.group_layout.addWidget(self.autofitCheck, 3, 1)

		self.originLabel = QtGui.QLabel("Auto Origin?")
		self.group_layout.addWidget(self.originLabel, 4, 0)

		self.originCheck = QtGui.QCheckBox()
		self.group_layout.addWidget(self.originCheck, 4, 1)

		self.group_layout.setColumnStretch(2,1)
		self.group.setLayout(self.group_layout)
		self.group.setStyleSheet(defaults.style_sheet)

		self.layout.addWidget(self.group)
		self.setLayout(self.layout)

	def getValues(self):
		out = []
		for widget in self.edits:
			out.append(int(widget.text()))
		return out

	def upload_origin(self, data):
		pid = 'Origin.ApplicationSI'
		origin = win32com.client.Dispatch(pid)

		short_name = "KRbInt1"
		long_name = "KRb Integrated"

		if origin.FindWorksheet(short_name) is None:
			origin.CreatePage(2, short_name, short_name)
		origin.Execute("{}!page.longname$ = {}".format(short_name, long_name))

		print data
		for (i, x) in enumerate(data):
			ret = origin.PutWorksheet("[{}]Sheet1".format(short_name), x, -1, i)

        # n = 0
        # for k in data:
        #     uploadSuccess = origin.PutWorksheet("[{}]Sheet1".format(short_name), k, -1, n)
        #     if uploadSuccess:
        #         n += 1
        #     else:
        #         print("Failed to upload to Origin. Is Sheet1 in the proper workbook available?")
        #         return -1