from PyQt4 import QtGui, QtCore, Qt
from PyQt4.QtCore import pyqtSignal
from twisted.internet.defer import inlineCallbacks
import twisted.internet.error

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle

from matplotlib import pyplot as plt

import time

import numpy as np
from copy import deepcopy

from krb_custom_colors import KRbCustomColors

import sys
sys.path.append('../')
import defaults


class Plot(QtGui.QWidget):
	def __init__(self, use_toolbar=True):
		super(Plot, self).__init__()
		self.use_toolbar = use_toolbar
		self.populate()

	# Populate GUI
	def populate(self):
		self.layout = QtGui.QVBoxLayout()

		self.figure = Figure()
		self.canvas = FigureCanvas(self.figure)

		if self.use_toolbar:
			self.toolbar = NavigationToolbar(self.canvas, self)
			self.layout.addWidget(self.toolbar)

		self.layout.addWidget(self.canvas)
		self.setLayout(self.layout)


def make_format_coord(numrows, numcols, data):
	def format_coord(x, y):
	    col = int(x + 0.5)
	    row = int(y + 0.5)
	    if col >= 0 and col < numcols and row >= 0 and row < numrows:
	        z = data[row, col]
	        return '({:},{:}), z={:.2f}'.format(int(x),int(y),z)
	    else:
	        return 'x=%1.4f, y=%1.4f' % (x, y)
	return format_coord


class Display(Plot):
	data = {}
	patch_parameters = [()]*4

	def __init__(self, dx, dy, use_toolbar=True, use_colorbar=True, title="", title_color='k'):
		super(Display, self).__init__(use_toolbar)
		self.setFixedSize(dx, dy)
		self.canvas.setContentsMargins(0,0,0,0)

		self.use_colorbar = use_colorbar

		self.title = title
		self.integrated_number = ""
		self.title_color = title_color

		# Colormaps
		self.colors = KRbCustomColors()
		self.cmaps = [self.colors.whiteJet, self.colors.whiteMagma, self.colors.whitePlasma, plt.cm.jet]

		self.ax = self.figure.add_subplot(111)
		self.frame = defaults.default_frame

		self.xlims = [0,200]
		self.ylims = [0,200]

		self.vlims = [0, 1]

	# Plot the data
	def replot(self):
		self.figure.clear()
		self.ax = self.figure.add_subplot(111)

		im = self.ax.imshow(
			self.data[self.frame],
			origin='upper',
			interpolation='none',
			vmin = self.vlims[0],
			vmax = self.vlims[1],
			cmap = self.cmaps[0])

		self.ax.set_xlim((self.xlims[0], self.xlims[1]))
		self.ax.set_ylim((self.ylims[1], self.ylims[0]))

		if self.use_colorbar:
			# Add a horizontal colorbar
			self.figure.colorbar(im, orientation='horizontal')

		# Don't understand patches... probably shouldn't mess with this
		for (i,pp) in enumerate(self.patch_parameters):
			if pp:
				p = Rectangle(*pp, linewidth=defaults.rect_linewidth, edgecolor=defaults.rect_colors[i], facecolor='none')
				self.ax.add_patch(p)

		# Need to do the following to get the z data to show up in the toolbar
		numrows, numcols = np.shape(self.data[self.frame])
		self.ax.format_coord = make_format_coord(numrows, numcols, self.data[self.frame])

		if self.title:
			self.ax.set_title(self.title + self.integrated_number, color=self.title_color)

		# Update the plot
		self.canvas.draw()

	def setROI(self, roi):
		self.xlims = [max(roi[0] - roi[2]/2, 0), min(roi[0] + roi[2]/2, defaults.dim_image[0])]
		self.ylims = [max(roi[1] - roi[3]/2, 0), min(roi[1] + roi[3]/2, defaults.dim_image[1])]

	def setVlims(self, vlims):
		self.vlims = vlims

	def setData(self, data):
		self.data = data

	def setFrame(self, frame):
		self.frame = frame

	def setIntegratedNumber(self, integrated_number):
		self.integrated_number = "{:.1f}".format(integrated_number)

	def addPatch(self, index, xy, w, h):
		self.patch_parameters[index] = (xy, w, h)


class Profile(Plot):
	data = {}

	def __init__(self):
		super(Profile, self).__init__(False)
		self.setFixedSize(defaults.dim_profile[0], defaults.dim_profile[1])
		self.xyRel = [0,0]

	def setData(self, data):
		self.data = data

	def setXYRel(self, xrel, yrel):
		self.xyRel = [xrel, yrel]

	# Plot the data
	def replot(self):
		# Clear plot
		self.figure.clear()

		# Plot the data
		ax = self.figure.add_subplot(111)
		for (xy,d) in zip(self.xyRel, self.data):
			start = int(xy-len(d)/2.0)
			t = np.arange(start, start+len(d), 1)
			ax.plot(t, d)
		self.canvas.draw()

class Profiles(QtGui.QWidget):
	n_rows = 4
	n_cols = 1

	def __init__(self):
		super(Profiles, self).__init__()
		self.setFixedSize(defaults.dim_profile_col[0], defaults.dim_profile_col[1])
		self.populate()

	def populate(self):
		self.layout = QtGui.QGridLayout()
		self.profiles = []
		for j in range(self.n_rows):
			for i in range(self.n_cols):
				profile = Profile()
				self.profiles.append(profile)
				self.layout.addWidget(profile, j, i)
		self.setLayout(self.layout)


class Zooms(QtGui.QWidget):
	n_rows = 4
	n_cols = 1

	def __init__(self):
		super(Zooms, self).__init__()
		self.setFixedSize(defaults.dim_zoom_col[0], defaults.dim_zoom_col[1])
		self.populate()

	def populate(self):
		self.layout = QtGui.QGridLayout()
		self.displays = []
		for j in range(self.n_rows):
			for i in range(self.n_cols):
				if j % 2 == 0:
					s = "Signal "
				else:
					s = "Background "
				s += defaults.state_names[j/2] + ": "

				display = Display(defaults.dim_zoom[0], defaults.dim_zoom[1], False, False, s, defaults.rect_colors[j*self.n_cols+i])
				self.displays.append(display)
				self.layout.addWidget(display, j, i)
		self.setLayout(self.layout)