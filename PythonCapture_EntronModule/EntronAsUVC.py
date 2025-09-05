# https://github.com/letmaik/pyvirtualcam
# Dependency : Need to install OBS, https://obsproject.com/
import pyvirtualcam

import os
import ctypes
import numpy as np
import cv2 as cv
import cv2
import time

def _draw_text(msg, render_image, origin, fs=None, fg=None, bg=None, THICK=None):
    FONT = cv2.FONT_HERSHEY_SIMPLEX
    AA = cv2.LINE_AA
    THICK = 1 if THICK is None else THICK
    PAD = 3
    fs = 0.45 if fs is None else fs * 0.45
    fg = (0, 0, 0) if fg is None else fg
    bg = (255, 255, 255) if bg is None else bg

    sz, baseline = cv2.getTextSize(msg, FONT, fs, THICK)
    cv2.rectangle(render_image, (origin[0] - PAD, origin[1] - sz[1] - PAD),
                  (origin[0] + sz[0] + PAD, origin[1] + sz[1] - baseline * 2 + PAD),
                  bg,
                  thickness=-1)
    cv2.putText(render_image, msg, origin, FONT, fs, fg, THICK, AA)

    return origin[1] + sz[1] + baseline + 1

#### Entron Module Constants
DLLName = "MODULE_SDK_ENTRON.dll"
# The following things won't be used but some values are required for the configure and start function
ZoneNbCode = 64
ZoneNb = 64
Resolution = 8
__YSize = 1344
__XSize = 1120
cTimeStamp = ctypes.c_uint16()
Enable_L8 = False

entronSDK = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),DLLName))

# Init the board
__YSize = 1344
__XSize = 1120

entronSDK = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),DLLName))

## Run the init function
platform = 1 # CX3
#platform = 2 # U96
module_version = 1 # 1 for L8 entron module, 2 for Zeekr module
entronSDK.module_init_with_path("temp", platform, Enable_L8, module_version)
print("Done Init")
bit_depth = 8
bin_mode = 0 # binning is either 0 or 2
target_order = 1
pre_strobe = 0
set_framerate = 30
# Stable Framerates @8bit w/o VL53L8 with Entron Module V1
# | Platform    | bin_mode  | Framerate |
# | CX3         | 0         | 30FPS     |
# | CX3         | 2         | 45FPS     |
# | U96         | 0         | 45FPS     |
# | U96         | 2         | 60FPS     |

# Start the stream
entronSDK.module_configure_and_start(ZoneNbCode, target_order, bit_depth, set_framerate, bin_mode, pre_strobe)
frameBufferSize = entronSDK.module_getVD56G4AbufferSize()
frameBufferData = ctypes.create_string_buffer(frameBufferSize)

#output_x = 800
#output_y = 700
bin_mode_dividing_factor_x = 1
bin_mode_dividing_factor_y = 1
if bin_mode == 2:
    bin_mode_dividing_factor_y = 2
    bin_mode_dividing_factor_x = 2
output_x = int(__XSize / bin_mode_dividing_factor_x)
output_y = int(__YSize / bin_mode_dividing_factor_y)
color = (0, 255, 0) #(B,G,R)
entronSDK.module_getImageData(frameBufferData, ctypes.byref(cTimeStamp))
previous_frame_time = time.time()
last_dropped_frame_time = time.time()
framerate_arr = []
average_framerate = 0
dropped_frame_count = 0
frame_num_last = cTimeStamp.value
enable_led = False
exposure_mode = 0
with pyvirtualcam.Camera(width=output_x, height=output_y, fps=set_framerate) as cam:
    print(f'Using virtual camera: {cam.device}')
    while True:
        try:
            entronSDK.module_getImageData(frameBufferData, ctypes.byref(cTimeStamp))
            frame_num_current = cTimeStamp.value
            if frame_num_last + 1 < frame_num_current:
                dropped_frame_count += frame_num_current - frame_num_last
                last_dropped_frame_time = time.time()
            frame_num_last = frame_num_current

            current_frame_time = time.time()
            framerate_arr.append(1 / (current_frame_time - previous_frame_time))
            previous_frame_time = current_frame_time
            if framerate_arr.__len__() > 10:
                average_framerate = np.average(framerate_arr)
                framerate_arr = []

            if bit_depth == 10:
                image = np.frombuffer(frameBufferData, dtype=np.uint16).reshape((output_y, output_x))
            else:
                image = np.frombuffer(frameBufferData, dtype=np.uint8).reshape((output_y, output_x))
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            cam.send(image)
            cam.sleep_until_next_frame()
            image = cv.resize(image, (0, 0), fx=0.5, fy=0.5)

            c, r = 2, 15
            r = _draw_text(f"Current framerate: {average_framerate:.2f} FPS", image, (c, r))
            r = _draw_text(f"{dropped_frame_count} frames dropped", image, (c, r))
            r = _draw_text(f"Time since last dropped frame: {(time.time() - last_dropped_frame_time):.1f}", image, (c, r))
            r = _draw_text(f"LED Status: {enable_led}", image,
                           (c, r))
            r = _draw_text(f"Exposure mode: {exposure_mode}", image,
                           (c, r))

            cv2.imshow('image stream', image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                entronSDK.module_stop()
                entronSDK.module_deinit()
                cv2.destroyAllWindows()
                exit()
            elif key == ord('l'):
                enable_led = not(enable_led)
                entronSDK.module_setIllumination(enable_led)
            elif key == ord('s'):
                exposure_mode = int(not(exposure_mode))
                entronSDK.module_setExposure(exposure_mode, 1,1)

        except Exception as err:
            entronSDK.module_stop()
            entronSDK.module_deinit()
            raise err