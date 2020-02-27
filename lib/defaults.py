import numpy as np

h_gui = 900
w_col = 400

dim_settings_col = (w_col, h_gui-100)
dim_display_col = (w_col, h_gui-100)
dim_zoom_col = (w_col, h_gui-100)
dim_profile_col = (w_col, h_gui-100)
dim_gui = (w_col*4, h_gui)

dim_roi = [w_col-50, 100]
h_load = 100
dim_analysis = (w_col/2, 200)
dim_display_opt = (w_col/2, 200)

dim_profile = (w_col, 200)
dim_zoom = (w_col, 200)

roi_default = [160, 210, 300, 500]
r00_signal_default = [160, 180, 60, 20]
r10_signal_default = [160, 220, 60, 20]
r00_background_default = [160, 200, 60, 20]
r10_background_default = [160, 240, 60, 20]

dim_image = [256, 512]

rect_linewidth = 2
rect_colors = ['b', 'k', 'r', 'y']

n_frames = 4

def fmapf(acq, fk):
	return 2*fk + acq
frame_map = {"shadow": fmapf(0,0), "light": fmapf(0,1), "dark": fmapf(1,0)}

frame_list = ["od", "shadow", "light", "dark"]
default_frame = "od"

default_path = "K:\\data\\{0.year}\\{0:%m}\\{0.year}{0:%m}{0:%d}\\Andor\\"

state_names = ["|0,0>", "|1,0>"]

######################
# Imaging parameters #
######################

binning = 1
pixel_size = 2.58e-6 * 2**binning

alpha = 2.0
CSatEff = (19000/4.0) * (2**binning)**2

sigma = 3 * (767e-9)**2 / (2 * np.pi)
Area = pixel_size**2

od_to_number = alpha * Area / sigma