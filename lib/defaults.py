import numpy as np
from PyQt4 import QtGui

h_gui = 900
w_col = 400

dim_settings_col = (w_col, h_gui-100)
dim_display_col = (w_col+100, h_gui-100)
dim_zoom_col = (w_col, h_gui-100)
dim_profile_col = (w_col, h_gui-100)
dim_gui = (w_col*4+50, h_gui)

dim_plot_group = (w_col*3, h_gui)

dim_roi = [w_col-50, 100]
h_load = 100
dim_analysis = (dim_settings_col[0]/2, 200)
dim_display_opt = (dim_settings_col[0]/2, 200)

dim_profile = (dim_profile_col[0], 200)
dim_zoom = (dim_zoom_col[0], 200)

roi_default = [150, 133, 100, 100]
r00_signal_default = [150, 133, 50, 20]
r10_signal_default = [150, 173, 50, 20]
r00_background_default = [150, 153, 50, 20]
r10_background_default = [150, 193, 50, 20]

dim_image = [512, 1024]

rect_linewidth = 2
oval_linewidth = 2
rect_colors = ['b', 'k', 'r', 'y']

style_sheet = "QGroupBox { font-weight: bold; } "

n_frames = 4

def fmapf(acq, fk):
	return 2*fk + acq
frame_map = {"shadow": fmapf(0,0), "light": fmapf(0,1), "dark": fmapf(1,0)}

frame_list = ["od", "shadow", "light", "dark"]
default_frame = "od"

default_path = "K:\\data\\{0.year}\\{0:%m}\\{0.year}{0:%m}{0:%d}\\KRbFK\\"

filebase = "krbfk_"
file_format = ".csv"
filename = filebase + "{}" + ".csv"

wait_for_load = 1 # seconds
autoload_loop = 2 # seconds


state_names = ["|0,0>", "|1,0>"]


fit_functions = ["Gaussian", "Gaussian w/ Gradient", "Fermi 2D", "Fermi 3D"]

fit_points = 10. # Points to plot per pixel

######################
# Imaging parameters #
######################

pixel_size = 2.58e-6

alpha = 2.0
c_sat_eff = (19000/4.0)

sigma = 3 * (767e-9)**2 / (2 * np.pi)
area = pixel_size**2

max_od = 10
od_to_number = alpha * area / sigma