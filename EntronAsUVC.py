# https://github.com/letmaik/pyvirtualcam
# Dependency : Need to install OBS, https://obsproject.com/
import pyvirtualcam

import os
import ctypes
import numpy as np
import cv2
import time
import multiprocessing

class EntronCamera():
    def __init__(self, platform="CX3", enable_L8 = False, display = True, module_version = 1, bit_depth = 8, bin_mode = 0, target_order = 1, pre_strobe = 0, set_framerate = 30):
        #### Entron Module Constants
        self.DLLName = "MODULE_SDK_ENTRON.dll"
        # The following things won't be used but some values are required for the configure and start function
        self.ZoneNbCode = 64
        self.ZoneNb = 64
        self.Resolution = 8
        self.__YSize = 1344
        self.__XSize = 1120
        self.cTimeStamp = ctypes.c_uint16()
        self.Enable_L8 = enable_L8

        self._DISPLAY = display

        self.entronSDK = ctypes.CDLL(os.path.join(os.path.dirname(os.path.abspath(__file__)),self.DLLName))

        ## Run the init function
        self.platform = 1 # CX3
        if platform == "U96":
            self.platform = 2
            
        self.module_version = module_version # 1 for L8 entron module, 2 for Zeekr module
        self.bit_depth = bit_depth
        self.bin_mode = bin_mode # binning is either 0 or 2
        self.target_order = target_order
        self.pre_strobe = pre_strobe
        self.set_framerate = set_framerate
        # Stable Framerates @8bit w/o VL53L8 with Entron Module V1
        # | Platform    | bin_mode  | Framerate |
        # | CX3         | 0         | 30FPS     |
        # | CX3         | 2         | 45FPS     |
        # | U96         | 0         | 45FPS     |
        # | U96         | 2         | 60FPS     |
        self.cam = None
        self.frameBufferData = None
        bin_mode_dividing_factor_x = 1
        bin_mode_dividing_factor_y = 1
        if self.bin_mode == 2:
            bin_mode_dividing_factor_y = 2
            bin_mode_dividing_factor_x = 2
        self.output_x = int(self.__XSize / bin_mode_dividing_factor_x)
        self.output_y = int(self.__YSize / bin_mode_dividing_factor_y)

    def _draw_text(msg, render_image, origin, fs=None, fg=None, bg=None, THICK=None):
        FONT = cv2.FONT_HERSHEY_SIMPLEX
        AA = cv2.LINE_AA
        THICK = 1 if THICK is None else THICK
        PAD = 3
        fs = 0.45 #if fs is None else fs * 0.45
        fg = (0, 0, 0) if fg is None else fg
        bg = (255, 255, 255) if bg is None else bg

        sz, baseline = cv2.getTextSize(msg, FONT, fs, THICK)
        cv2.rectangle(render_image, (origin[0] - PAD, origin[1] - sz[1] - PAD),
                    (origin[0] + sz[0] + PAD, origin[1] + sz[1] - baseline * 2 + PAD),
                    bg,
                    thickness=-1)
        cv2.putText(render_image, msg, origin, FONT, fs, fg, THICK, AA)

        return origin[1] + sz[1] + baseline + 1

    def initialize_and_start_module(self):
        try:
            self.entronSDK.module_init_with_path("temp", self.platform, self.Enable_L8, self.module_version)
            print("Done Init")
        
            # Start the stream
            self.entronSDK.module_configure_and_start(self.ZoneNbCode, self.target_order, self.bit_depth, self.set_framerate, self.bin_mode, self.pre_strobe)
            frameBufferSize = self.entronSDK.module_getVD56G4AbufferSize()
            self.frameBufferData = ctypes.create_string_buffer(frameBufferSize)
            self.cam = pyvirtualcam.Camera(width=self.output_x, height=self.output_y, fps=self.set_framerate)
            print(f'Using virtual camera: {self.cam.device}')
            return True
        except Exception as e:
            print(f"Error in initializing camera: {e}")
            return False

    def stream_virtual_cam(self, end_event):
        def _draw_text(msg, render_image, origin, fs=None, fg=None, bg=None, THICK=None):
            FONT = cv2.FONT_HERSHEY_SIMPLEX
            AA = cv2.LINE_AA
            THICK = 1 if THICK is None else THICK
            PAD = 3
            fs = 0.45 #if fs is None else fs * 0.45
            fg = (0, 0, 0) if fg is None else fg
            bg = (255, 255, 255) if bg is None else bg

            sz, baseline = cv2.getTextSize(msg, FONT, fs, THICK)
            cv2.rectangle(render_image, (origin[0] - PAD, origin[1] - sz[1] - PAD),
                        (origin[0] + sz[0] + PAD, origin[1] + sz[1] - baseline * 2 + PAD),
                        bg,
                        thickness=-1)
            cv2.putText(render_image, msg, origin, FONT, fs, fg, THICK, AA)

            return origin[1] + sz[1] + baseline + 1
        color = (0, 255, 0) #(B,G,R)
        self.entronSDK.module_getImageData(self.frameBufferData, ctypes.byref(self.cTimeStamp))
        previous_frame_time = time.time()
        last_dropped_frame_time = time.time()
        framerate_arr = []
        average_framerate = 0
        dropped_frame_count = 0
        frame_num_last = self.cTimeStamp.value
        enable_led = False
        exposure_mode = 0
        while True:
            try:
                self.entronSDK.module_getImageData(self.frameBufferData, ctypes.byref(self.cTimeStamp))
                frame_num_current = self.cTimeStamp.value
                if frame_num_last + 1 < frame_num_current:
                    dropped_frame_count += frame_num_current - frame_num_last
                    last_dropped_frame_time = time.time()
                frame_num_last = frame_num_current

                current_frame_time = time.time()
                if current_frame_time != previous_frame_time:
                    framerate_arr.append(1 / (current_frame_time - previous_frame_time))
                    previous_frame_time = current_frame_time
                    if framerate_arr.__len__() > 10:
                        average_framerate = np.average(framerate_arr)
                        framerate_arr = []

                if self.bit_depth == 10:
                    image = np.frombuffer(self.frameBufferData, dtype=np.uint16).reshape((self.output_y, self.output_x))
                else:
                    image = np.frombuffer(self.frameBufferData, dtype=np.uint8).reshape((self.output_y, self.output_x))
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                self.cam.send(image)
                self.cam.sleep_until_next_frame()
                if os.path.exists(os.path.join(os.path.abspath("."), "1")):
                    os.remove(os.path.join(os.path.abspath("."), "1"))
                    exposure_mode = 1
                    self.entronSDK.module_setExposure(exposure_mode, 1,1)
                if os.path.exists(os.path.join(os.path.abspath("."), "0")):
                    os.remove(os.path.join(os.path.abspath("."), "0"))
                    exposure_mode = 0
                    self.entronSDK.module_setExposure(exposure_mode, 1,1)
                if self._DISPLAY:
                    image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

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
                    if (key == ord('q') or end_event.is_set()):
                        self.entronSDK.module_stop()
                        self.entronSDK.module_deinit()
                        cv2.destroyAllWindows()
                        exit()
                    elif key == ord('l'):
                        enable_led = not(enable_led)
                        self.entronSDK.module_setIllumination(enable_led)
                    elif key == ord('s'):
                        exposure_mode = int(not(exposure_mode))
                        self.entronSDK.module_setExposure(exposure_mode, 1,1)

            except Exception as err:
                self.entronSDK.module_stop()
                self.entronSDK.module_deinit()
                raise err

def make_and_stream_camera(good_event, bad_event, end_event):
    entron = EntronCamera()
    if entron.initialize_and_start_module():
        good_event.set()
        entron.stream_virtual_cam(end_event)
    bad_event.set()

if __name__ == "__main__":
    entron = EntronCamera()
    entron.initialize_and_start_module()
    entron.stream_virtual_cam()
    