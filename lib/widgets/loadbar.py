from PyQt4 import QtGui, QtCore, Qt
from PyQt4.QtCore import pyqtSignal
from twisted.internet.defer import inlineCallbacks
import twisted.internet.error

import time
import os

import numpy as np
from copy import deepcopy

import datetime

import sys
sys.path.append('../')
import defaults

class LoadBar(QtGui.QWidget):
	loadSignal = pyqtSignal()
	autoloadSignal = pyqtSignal(bool)

	def __init__(self, args=None):
		super(LoadBar, self).__init__()
		self.setFixedSize(defaults.w_col, defaults.h_load)
		self.initialize()
		self.populate()

	def initialize(self):
		self.pathLabel = QtGui.QLabel("Path")

		p = defaults.default_path.format(datetime.datetime.now())
		self.pathEdit = QtGui.QLineEdit(p)
		filenumber = self.lastFile(p)

		self.loadButton = QtGui.QPushButton("Load")
		self.loadButton.clicked.connect(self.load)
		self.browseButton = QtGui.QPushButton("Browse")
		self.browseButton.clicked.connect(self.browse)

		self.fileLabel = QtGui.QLabel("File #")
		self.fileEdit = QtGui.QLineEdit(str(filenumber))

		self.autoLabel = QtGui.QLabel("Autoload?")
		self.autoCheck = QtGui.QCheckBox()
		self.autoCheck.stateChanged.connect(self.autoload)

	def populate(self):
		self.layout = QtGui.QHBoxLayout()
		self.group = QtGui.QGroupBox("Load file")
		self.group_layout = QtGui.QVBoxLayout()

		gl1 = QtGui.QHBoxLayout()
		gl2 = QtGui.QHBoxLayout()

		gl1.addWidget(self.pathLabel)
		gl1.addWidget(self.pathEdit)
		gl1.addWidget(self.browseButton)
		gl1.addWidget(self.loadButton)
		
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

	def lastFile(self, p):
		# Check if the directory exists
		if os.path.isdir(p):
			filelist = os.listdir(p)
			filelist.sort(key=lambda x: int(x.split(defaults.filebase)[-1].split(defaults.file_format)[0]))

			if not filelist:
				return 0
			last = filelist[-1]
			
			if last.find(defaults.filebase) == 0:
				return int(last.split(defaults.filebase)[-1].split(defaults.file_format)[0])
			else:
				return 0
		return 0

	def getFileNumber(self):
		return int(self.fileEdit.text())

	def getPath(self):
		return self.pathEdit.text()

	def setFilePath(self, path):
		self.pathEdit.setText(path)

	def setFileNumber(self, number):
		self.fileEdit.setText(str(int(number)))

	def browse(self):
		now = datetime.datetime.now()
		path = defaults.default_path.format(now)
		filename = QtGui.QFileDialog.getOpenFileName(self,
    		"Open Image", path, "Image Files (*.csv)")

		if filename:
			self.setFilePath(filename)

	# Requires a higher level action
	def load(self):
		self.loadSignal.emit()

	def autoload(self):
		if self.autoCheck.isChecked():
			self.autoloadSignal.emit(True)
		else:
			self.autoloadSignal.emit(False)