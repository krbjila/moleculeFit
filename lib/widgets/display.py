from PyQt4 import QtGui, QtCore, Qt
from PyQt4.QtCore import pyqtSignal
from twisted.internet.defer import inlineCallbacks
import twisted.internet.error

from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Ellipse
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from matplotlib.gridspec import GridSpec


from matplotlib import pyplot as plt

import time

import numpy as np
from copy import deepcopy

from krb_custom_colors import KRbCustomColors

import sys
sys.path.append('../')
import defaults

class PlotGroup(QtGui.QWidget):
	crosshairs_placed = pyqtSignal(float, float)

	grid_size = 4

	template = {
		"ax": None,
		"rects": [None]*grid_size,
		"ovals": [],
		"cross": (),
		"crossColors": ("C0", "C1"),
		"lims": ((), ()),
		"frame": "od",
		"data": {},
		"title": "",
		"number": 0,
		"color": "",
		"colorbar": False,
		"clickable": False,
		"xyRel": (0,0),
	}


	def __init__(self):
		super(PlotGroup, self).__init__()

		# Colormaps
		self.colors = KRbCustomColors()
		self.cmaps = [self.colors.whiteJet, self.colors.whiteMagma, self.colors.whitePlasma, plt.cm.jet]
		self.colorbar = 0

		self.vlims = [0, 1.0]

		self.main = deepcopy(self.template)
		self.main["colorbar"] = True
		self.main["clickable"] = True
		self.main["crossColors"] = ("k", "k")

		self.displays = [deepcopy(self.template) for i in range(self.grid_size)]
		self.profiles = [deepcopy(self.template) for i in range(self.grid_size)]

		self.dict = {"m": self.main, "d": self.displays, "p": self.profiles}

		self.populate()

	def populate(self):
		self.layout = QtGui.QVBoxLayout()

		self.figure = Figure(frameon=False, tight_layout=True)
		self.canvas = FigureCanvas(self.figure)
		self.canvas.mpl_connect('button_press_event', self.clicked)

		gridspec = GridSpec(self.grid_size, self.grid_size, figure=self.figure)

		self.main["ax"] = self.figure.add_subplot(gridspec[:, 0:(self.grid_size-2)])
		for i in range(self.grid_size):
			s1 = "Signal " if i % 2 == 0 else "Background "
			s2 = defaults.state_names[i/2]

			self.displays[i]["ax"] = self.figure.add_subplot(gridspec[i, self.grid_size-2])
			self.displays[i]["title"] = s1 + s2 + ": "
			self.displays[i]["color"] = defaults.rect_colors[i]

			self.profiles[i]["ax"] = self.figure.add_subplot(gridspec[i, self.grid_size-1])
			
		self.layout.addWidget(self.canvas)
		self.setLayout(self.layout)

	def clicked(self, event):
		if self.main["data"]:
			# On double-click, remove crosshairs
			if event.dblclick:
				for w in [self.main] + self.displays + self.profiles:
					if w["ax"] == event.inaxes and w["clickable"]:
						w["cross"] = ()
						self.replot()
			# On single-clikc, place crosshairs
			else:
				for w in [self.main] + self.displays + self.profiles:
					if w["ax"] == event.inaxes and w["clickable"]:
						w["cross"] = (event.xdata, event.ydata)
						self.replot()
						self.crosshairs_placed.emit(event.xdata, event.ydata)

	def setVlims(self, vlims):
		self.vlims = vlims

	def replot(self):
		self.replotImage(self.main)
		for (d,p) in zip(self.displays, self.profiles):
			self.replotImage(d)
			self.replotProfile(p)
		self.canvas.draw()

	def lookup(self, flag, index):
		if flag == "m":
			return self.dict[flag]
		elif flag == "d" or flag == "p":
			return self.dict[flag][index]

	def setData(self, flag, index, data):
		d = self.lookup(flag, index)
		d["data"] = data

	def setFrame(self, flag, index, frame):
		d = self.lookup(flag, index)
		d["frame"] = frame

	def setNumber(self, flag, index, number):
		d = self.lookup(flag, index)
		d["number"] = number

	def setROI(self, flag, index, roi):
		xlims = (max(roi[0] - roi[2]/2, 0), min(roi[0] + roi[2]/2, defaults.dim_image[0]))
		ylims = (min(roi[1] + roi[3]/2, defaults.dim_image[1]), max(roi[1] - roi[3]/2, 0))

		self.setLims(flag, index, xlims, ylims)

	# As tuples
	def setLims(self, flag, index, xlims, ylims):
		d = self.lookup(flag, index)
		d["lims"] = (xlims, ylims)

	def setCross(self, flag, index, xc, yc):
		d = self.lookup(flag, index)
		d["cross"] = (xc, yc)

	def setRectMain(self, index, xy, width, height, color):
		self.main["rects"][index] = {
			"xy": xy,
			"w": width,
			"h": height,
			"c": color
		}

	def setOvalDisplay(self, index, xy, width, height, color):
		self.displays[index]["ovals"] = [{
			"xy": xy,
			"w": width,
			"h": height,
			"c": color
		}]

	def setXYRelProfile(self, index, xyrel):
		self.profiles[index]["xyRel"] = xyrel

	def replotImage(self, properties):
		ax = properties["ax"]
		frame = properties["frame"]
		data = properties["data"][frame]
		lims = properties["lims"]
		rects = properties["rects"]
		ovals = properties["ovals"]
		cross = properties["cross"]
		crossColors = properties["crossColors"]
		title = properties["title"]
		color = properties["color"]
		number = properties["number"]
		colorbar = properties["colorbar"]

		ax.clear()

		im = ax.imshow(
			data,
			origin='upper',
			interpolation='none',
			vmin = self.vlims[0],
			vmax = self.vlims[1],
			cmap = self.cmaps[0])

		ax.set_xlim(lims[0])
		ax.set_ylim(lims[1])

		if colorbar:
			if self.colorbar:
				self.colorbar.remove()

			# https://stackoverflow.com/a/18195921
			divider = make_axes_locatable(ax)
			cax = divider.append_axes("right", size="5%", pad=0.05)
			self.colorbar = self.figure.colorbar(im, cax=cax, orientation='vertical')

		for r in rects:
			if r:
				p = Rectangle(r["xy"], r["w"], r["h"], linewidth=defaults.rect_linewidth, edgecolor=r["c"], facecolor='none')
				ax.add_patch(p)

		for o in ovals:
			if o:
				p = Ellipse(o["xy"], o["w"], o["h"], linewidth=defaults.oval_linewidth, edgecolor=o["c"], facecolor='none')
				ax.add_patch(p)

		if cross:
			xline = ((0, defaults.dim_image[0]), (cross[1], cross[1]))
			yline = ((cross[0], cross[0]), (0, defaults.dim_image[1]))
			ax.add_line(Line2D(xline[0], xline[1], color=crossColors[0]))
			ax.add_line(Line2D(yline[0], yline[1], color=crossColors[1]))


		# # Need to do the following to get the z data to show up in the toolbar
		# numrows, numcols = np.shape(self.data[self.frame])
		# self.ax.format_coord = make_format_coord(numrows, numcols, self.data[self.frame])

		if title and color:
			ax.set_title(title + "N = {:.1f}".format(number), color=color)

	def replotProfile(self, properties):
		ax = properties["ax"]
		xyRel = properties["xyRel"]
		data = properties["data"]
		title = properties["title"]
		color = properties["color"]

		ax.clear()

		for (xy,d,c) in zip(xyRel, data, ['C0', 'C1']):
			start = int(xy-len(d)/2.0)
			t = np.arange(start, start+len(d), 1)
			ax.plot(t, d)

		if title and color:
			ax.set_title(title, color=c)

	


# class Plot(QtGui.QWidget):
# 	def __init__(self, use_toolbar=True):
# 		super(Plot, self).__init__()
# 		self.use_toolbar = use_toolbar
# 		self.populate()

# 	# Populate GUI
# 	def populate(self):
# 		self.layout = QtGui.QVBoxLayout()

# 		self.figure = Figure()
# 		self.canvas = FigureCanvas(self.figure)

# 		if self.use_toolbar:
# 			self.toolbar = NavigationToolbar(self.canvas, self)
# 			self.layout.addWidget(self.toolbar)

# 		self.layout.addWidget(self.canvas)
# 		self.setLayout(self.layout)


# def make_format_coord(numrows, numcols, data):
# 	def format_coord(x, y):
# 	    col = int(x + 0.5)
# 	    row = int(y + 0.5)
# 	    if col >= 0 and col < numcols and row >= 0 and row < numrows:
# 	        z = data[row, col]
# 	        return '({:},{:}), z={:.2f}'.format(int(x),int(y),z)
# 	    else:
# 	        return 'x=%1.4f, y=%1.4f' % (x, y)
# 	return format_coord


# class Display(Plot):
# 	data = {}
# 	patch_parameters = [()]*4

# 	xline = ((), ())
# 	yline = ((), ())
# 	oval = {'xy': (0,0), 'w': 0, 'h': 0}

# 	def __init__(self, dx, dy, use_toolbar=True, use_colorbar=True, title="", title_color='k'):
# 		super(Display, self).__init__(use_toolbar)
# 		self.setFixedSize(dx, dy)
# 		self.canvas.setContentsMargins(0,0,0,0)

# 		self.use_colorbar = use_colorbar

# 		self.title = title
# 		self.integrated_number = ""
# 		self.title_color = title_color

# 		# Colormaps
# 		self.colors = KRbCustomColors()
# 		self.cmaps = [self.colors.whiteJet, self.colors.whiteMagma, self.colors.whitePlasma, plt.cm.jet]

# 		self.ax = self.figure.add_subplot(111)
# 		self.frame = defaults.default_frame

# 		self.xlims = [0,200]
# 		self.ylims = [0,200]

# 		self.vlims = [0, 1]

# 	# Plot the data
# 	def replot(self):
# 		self.figure.clear()
# 		self.ax = self.figure.add_subplot(111)

# 		im = self.ax.imshow(
# 			self.data[self.frame],
# 			origin='upper',
# 			interpolation='none',
# 			vmin = self.vlims[0],
# 			vmax = self.vlims[1],
# 			cmap = self.cmaps[0])

# 		self.ax.set_xlim((self.xlims[0], self.xlims[1]))
# 		self.ax.set_ylim((self.ylims[1], self.ylims[0]))

# 		if self.use_colorbar:
# 			# https://stackoverflow.com/a/18195921
# 			divider = make_axes_locatable(self.ax)
# 			cax = divider.append_axes("right", size="5%", pad=0.05)
# 			# Add a horizontal colorbar
# 			self.figure.colorbar(im, cax=cax, orientation='vertical')

# 		# Don't understand patches... probably shouldn't mess with this
# 		for (i,pp) in enumerate(self.patch_parameters):
# 			if pp:
# 				p = Rectangle(*pp, linewidth=defaults.rect_linewidth, edgecolor=defaults.rect_colors[i], facecolor='none')
# 				self.ax.add_patch(p)

# 		self.ax.add_line(Line2D(self.xline[0], self.xline[1], color='C0'))
# 		self.ax.add_line(Line2D(self.yline[0], self.yline[1], color='C1'))

# 		self.ax.add_patch(
# 			Ellipse(
# 				self.oval['xy'],
# 				self.oval['w'],
# 				self.oval['h'],
# 				edgecolor=self.title_color,
# 				facecolor='none')
# 			)

# 		# Need to do the following to get the z data to show up in the toolbar
# 		numrows, numcols = np.shape(self.data[self.frame])
# 		self.ax.format_coord = make_format_coord(numrows, numcols, self.data[self.frame])

# 		if self.title:
# 			self.ax.set_title(self.title + "N = {:.1f}".format(self.integrated_number), color=self.title_color)

# 		# Update the plot
# 		self.canvas.draw()

# 	def setROI(self, roi):
# 		self.xlims = [max(roi[0] - roi[2]/2, 0), min(roi[0] + roi[2]/2, defaults.dim_image[0])]
# 		self.ylims = [min(roi[1] + roi[3]/2, defaults.dim_image[1]), max(roi[1] - roi[3]/2, 0)]

# 	def setVlims(self, vlims):
# 		self.vlims = vlims

# 	def setData(self, data):
# 		self.data = data

# 	def setFrame(self, frame):
# 		self.frame = frame

# 	def setCross(self, xline, yline):
# 		self.xline = xline
# 		self.yline = yline

# 	def setOval(self, xy, width, height):
# 		self.oval['xy'] = xy
# 		self.oval['w'] = width
# 		self.oval['h'] = height

# 	def setIntegratedNumber(self, integrated_number):
# 		self.integrated_number = integrated_number

# 	def addPatch(self, index, xy, w, h):
# 		self.patch_parameters[index] = (xy, w, h)


# class Profile(Plot):
# 	data = {}

# 	def __init__(self):
# 		super(Profile, self).__init__(False)
# 		self.setFixedSize(defaults.dim_profile[0], defaults.dim_profile[1])
# 		self.xyRel = [0,0]

# 	def setData(self, data):
# 		self.data = data

# 	def setXYRel(self, xrel, yrel):
# 		self.xyRel = [xrel, yrel]

# 	# Plot the data
# 	def replot(self):
# 		# Clear plot
# 		self.figure.clear()

# 		# Plot the data
# 		ax = self.figure.add_subplot(111)
# 		for (xy,d) in zip(self.xyRel, self.data):
# 			start = int(xy-len(d)/2.0)
# 			t = np.arange(start, start+len(d), 1)
# 			ax.plot(t, d)
# 		self.canvas.draw()

# class Profiles(QtGui.QWidget):
# 	n_rows = 4
# 	n_cols = 1

# 	def __init__(self):
# 		super(Profiles, self).__init__()
# 		self.setFixedSize(defaults.dim_profile_col[0], defaults.dim_profile_col[1])
# 		self.populate()

# 	def populate(self):
# 		self.layout = QtGui.QGridLayout()
# 		self.profiles = []
# 		for j in range(self.n_rows):
# 			for i in range(self.n_cols):
# 				profile = Profile()
# 				self.profiles.append(profile)
# 				self.layout.addWidget(profile, j, i)
# 		self.setLayout(self.layout)


# class Zooms(QtGui.QWidget):
# 	n_rows = 4
# 	n_cols = 1

# 	def __init__(self):
# 		super(Zooms, self).__init__()
# 		self.setFixedSize(defaults.dim_zoom_col[0], defaults.dim_zoom_col[1])
# 		self.populate()

# 	def populate(self):
# 		self.layout = QtGui.QGridLayout()
# 		self.displays = []
# 		for j in range(self.n_rows):
# 			for i in range(self.n_cols):
# 				if j % 2 == 0:
# 					s = "Signal "
# 				else:
# 					s = "Background "
# 				s += defaults.state_names[j/2] + ": "

# 				display = Display(defaults.dim_zoom[0], defaults.dim_zoom[1], False, False, s, defaults.rect_colors[j*self.n_cols+i])
# 				self.displays.append(display)
# 				self.layout.addWidget(display, j, i)
# 		self.setLayout(self.layout)