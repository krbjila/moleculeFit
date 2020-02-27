from PyQt4 import QtGui, QtCore, Qt
from PyQt4.QtCore import pyqtSignal
from twisted.internet.defer import inlineCallbacks
import twisted.internet.error

import time

import numpy as np
from copy import deepcopy

import datetime

import sys
sys.path.append('../')
import defaults

class LoadBar(QtGui.QWidget):
	loadSignal = pyqtSignal()

	def __init__(self, args=None):
		super(LoadBar, self).__init__()
		self.setFixedSize(defaults.w_col, defaults.h_load)
		self.populate()

	def populate(self):
		self.layout = QtGui.QHBoxLayout()
		self.group = QtGui.QGroupBox("Load file")
		self.group_layout = QtGui.QVBoxLayout()

		gl1 = QtGui.QHBoxLayout()
		gl2 = QtGui.QHBoxLayout()

		self.pathLabel = QtGui.QLabel("Path")
		self.pathEdit = QtGui.QLineEdit()

		self.loadButton = QtGui.QPushButton("Load")
		self.loadButton.clicked.connect(self.load)
		self.browseButton = QtGui.QPushButton("Browse")
		self.browseButton.clicked.connect(self.browse)

		gl1.addWidget(self.pathLabel)
		gl1.addWidget(self.pathEdit)
		gl1.addWidget(self.browseButton)
		gl1.addWidget(self.loadButton)

		self.fileLabel = QtGui.QLabel("File #")
		self.fileEdit = QtGui.QLineEdit()

		self.autoLabel = QtGui.QLabel("Autoload?")
		self.autoCheck = QtGui.QCheckBox()

		gl2.addStretch(1)
		gl2.addWidget(self.fileLabel)
		gl2.addWidget(self.fileEdit)
		gl2.addWidget(self.autoLabel)
		gl2.addWidget(self.autoCheck)

		self.group_layout.addLayout(gl1)
		self.group_layout.addLayout(gl2)
		self.group.setLayout(self.group_layout)
		self.group.setStyleSheet(defaults.style_sheet)

		self.layout.addWidget(self.group)

		self.setLayout(self.layout)

	def browse(self):
		now = datetime.datetime.now()
		path = defaults.default_path.format(now)
		filename = QtGui.QFileDialog.getOpenFileName(self,
    		"Open Image", path, "Image Files (*.csv)")

		if filename:
			self.pathEdit.setText(filename)

	# Requires a higher level action
	def load(self):
		self.loadSignal.emit()

class Autoloader(object):
	def __init__(self):
		print "hi"

	def load(self, path):
		return np.loadtxt(path, delimiter=',')