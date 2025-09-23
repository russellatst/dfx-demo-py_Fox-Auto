###################################################
# Author: Russell Wong
# TODO: add camera control
# TODO: make user interface
###################################################
import os.path
import cv2
import dfxdemo.dfxdemo
import multiprocessing
from multiprocessing import Process, shared_memory
import numpy as np
import asyncio
import time
import enum

start_str = "start"
process_limit_str = "process_limit"
measuring_in_progress_str = "measuring_in_progress"
error_str = "error"
end_process_str = "end"
initialzied_txt = "initialized"
message_dir = os.path.join(os.path.abspath("."), "messages")
cam = cv2.VideoCapture(1)

class app_state(enum.Enum):
            INITIALIZING = 0
            STANDBY = 1,
            MEASURING = 2,
            LIMIT_REACHED = 3,
            TERMINATED = 4

class HRM_Proc():
    def __init__(self, status_shm, hr_shm, coordinator_shm, start_event, status_event, hr_event,app_num):
        
        self.status_shm = status_shm
        self.hr_shm = hr_shm
        self.start_event = start_event
        self.status_event = status_event
        self.hr_event = hr_event
        self.app_num = app_num
        self.proc = None
        self.coordinator_shm = coordinator_shm
        self.current_state = app_state.TERMINATED
    
    def run_worker(self,seconds_to_wait_before_starting):
        json_config_file = "config1.json"
        cam_num = 1
        if self.app_num ==2:
            json_config_file = "config2.json"
        try:
            asyncio.run(dfxdemo.dfxdemo.run_measurements(json_config_file,1,120, self.app_num, self.status_shm, self.hr_shm, self.start_event, self.status_event, self.hr_event, self.coordinator_shm,seconds_to_wait_before_starting))
            self.current_state = app_state.INITIALIZING
        except Exception as e:
            raise e

    def start_process(self, seconds_to_wait_before_starting):
        print(f"Starting PROC {self.app_num}")
        try:
            self.proc = Process(target=self.run_worker, name=str(self.app_num), args=(seconds_to_wait_before_starting,))
            self.proc.start()
            self.start_event.clear()
            print(f"Successful start of PROC {self.app_num}")
        except Exception as e:
            print(f"!!!!! Failure starting PROC {self.app_num}")
            raise e
        return self.proc
    
    def stop_process(self):
        print(f"ENDING PROC {self.app_num}")
        try:
            self.write_shm_message(self.coordinator_shm, self.start_event, end_process_str)
            self.start_event.set()
            if self.proc != None:
                self.proc.terminate()
                self.proc = None
            self.current_state = app_state.TERMINATED
            print(f"Successful end of PROC {self.app_num}")
        except Exception as e:
            print(f"!!!!! Failure ending PROC {self.app_num}")
            raise e

    def write_shm_message(self, shm, event, text):
        shm.buf[:text.__len__()] = bytearray(text, 'utf-8')
        event.set()

    def get_hr_message(self):
        s = str(bytes(self.hr_shm.buf[:]).decode()).replace("\x00", "")
        self.hr_shm.buf[:] = bytearray(self.hr_shm.buf.nbytes)
        return s
    
    def get_status_message(self):
        s = str(bytes(self.status_shm.buf[:]).decode()).replace("\x00", "")
        self.status_shm.buf[:] = bytearray(self.status_shm.buf.nbytes)
        return s
    
    def get_message(self, shm):
        s = str(bytes(shm.buf[:]).decode()).replace("\x00", "")
        shm.buf[:] = bytearray(shm.buf.nbytes)
        return s


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

def set_active_process(app_num):
    active_app = app_num
    inactive_app = (app_num % 2) + 1
    print(f"New active: {active_app}")
    print(f"New Inactive: {inactive_app}")
    return active_app, inactive_app

def display_result(app_num):
    ret, frame = cam.read()
    c, r = 2, 15
    r = _draw_text(f"Current Result: {app_num}", frame, (c,r))
    cv2.imshow("Result", frame)
    k = cv2.waitKey(1) & 0xFF
    return k

def write_shm_message(shm, event, text):
    shm.buf[:text.__len__()] = bytearray(text, 'utf-8')
    event.set()

class coordinator_state(enum.Enum):
            INITIALIZING = 0
            ACTIVE_MEASURING = 1,
            START_INACTIVE = 2,
            WAITING_FOR_USER= 3

def manage_hr_process(worker_arr, hr_shm_gui, hr_event_gui):
    active_app = 1
    inactive_app = 2
    worker_arr[active_app - 1].start_process(2)
    worker_arr[inactive_app - 1].start_process(2)

    state = coordinator_state.INITIALIZING
    while True:
        if worker_arr[active_app - 1].status_event.is_set():
            message_str = worker_arr[active_app - 1].get_status_message()
            print(f"Active app ({active_app}) message received: {message_str}")
            print(f"Active app {active_app} state: {state}")
            if message_str == process_limit_str and state == coordinator_state.ACTIVE_MEASURING:
                print(f"Limit received from {active_app}")
                worker_arr[active_app - 1].current_state = app_state.LIMIT_REACHED
                print("Starting inactive")
                worker_arr[inactive_app - 1].start_event.set()
                worker_arr[active_app - 1].start_event.clear()
                print(f"Starting heart rate for proc: {inactive_app}")
                state = coordinator_state.START_INACTIVE
            elif message_str == initialzied_txt and state == coordinator_state.INITIALIZING:
                worker_arr[active_app - 1].start_event.set()
                state = coordinator_state.WAITING_FOR_USER
                print(f"Worker started: {active_app}")
            elif message_str == measuring_in_progress_str:
                write_shm_message(hr_shm_gui, hr_event_gui,"Finding subject...")
                print("Measuring in progress started.")
                state = coordinator_state.ACTIVE_MEASURING
            elif message_str == error_str:
                    print(f"ERROR FROM ACTIVE APP {active_app}")
                    return 0
            worker_arr[active_app - 1].status_event.clear()

        if worker_arr[inactive_app - 1].status_event.is_set():
            message_str = worker_arr[inactive_app - 1].get_status_message()
            print(f"Inactive app ({inactive_app}) message received: {message_str}")
            print(f"Inactive app {inactive_app} state: {state}")
            if message_str == measuring_in_progress_str and state == coordinator_state.START_INACTIVE:
                print("Swapping processes")
                worker_arr[active_app - 1].stop_process()
                active_app, inactive_app = set_active_process(inactive_app)
                worker_arr[inactive_app - 1].start_process(2)
                state = coordinator_state.ACTIVE_MEASURING
            elif message_str == error_str:
                    print(f"ERROR FROM INACTIVE APP {inactive_app}")
                    return 0
            worker_arr[inactive_app - 1].status_event.clear()

        if worker_arr[active_app - 1].hr_event.is_set():
            hr_str = worker_arr[active_app - 1].get_hr_message()
            if int(hr_str[0]) > 1:
                hr_str = hr_str[0:2] + "." + hr_str[2:]
            else:
                hr_str = hr_str[0:3] + "." + hr_str[3:]
            print(f"Active HR is {hr_str}")
            write_shm_message(hr_shm_gui,hr_event_gui,hr_str)
            if(hr_event_gui.is_set()):
                 print("hooray, set")
            worker_arr[active_app - 1].hr_event.clear()
             
        time.sleep(1)

def gui_process(hr_shm, hr_event, status_shm, status_ready):
    output_x = 2560
    output_y = 1440
    frame_offset_y = 50
    frame_offset_x = 50
    FONT = cv2.FONT_HERSHEY_PLAIN
    AA = cv2.LINE_AA
    THICK = 4
    fs = 3
    fg = (255, 255, 255)
    blank_frame = np.zeros((output_y, output_x, 3), dtype=np.uint8)
    st_logo = cv2.imread(os.path.join(os.path.abspath("."),"assets","ST_logo_2024_black.png"))
    st_logo = cv2.resize(st_logo, (0,0), fx= 0.4, fy=0.4, interpolation=cv2.INTER_AREA)
    project_banner = cv2.imread(os.path.join(os.path.abspath("."),"assets","Project-Banner.png"))
    nuralogix_logo = cv2.imread(os.path.join(os.path.abspath("."),"assets","nura_black.png"))
    
    class gui_state(enum.Enum):
        INITIALIZING = 0,
        WAITING_FOR_SUBJECT = 1,
        ANALYZING = 2,
        HEART_RATE_MONITORING_IN_PROGRESS = 3

    def write_shm_message(shm, event, text):
        shm.buf[:text.__len__()] = bytearray(text, 'utf-8')
        event.set()

    def get_hr_message():
        s = str(bytes(hr_shm.buf[:]).decode()).replace("\x00", "")
        print(f"FROM IN GUI: {s}")
        hr_shm.buf[:] = bytearray(hr_shm.buf.nbytes)
        hr_event.clear()
        return s
    
    def get_status_message():
        s = str(bytes(status_shm.buf[:]).decode()).replace("\x00", "")
        status_shm.buf[:] = bytearray(status_shm.buf.nbytes)
        return s
    
    def draw_frame(live_image,gstate):
        green = (0, 255, 0)
        yellow = (104, 232, 2444)
        white = (255, 255, 255)
        red = (0, 0, 255)
        display_frame[frame_offset_y : frame_offset_y + live_image.shape[0], 
                       frame_offset_x : frame_offset_x + live_image.shape[1],
                       :] = live_image
        if gstate == gui_state.INITIALIZING:
            text = "Initializing"
            color = yellow
        elif gstate == gui_state.WAITING_FOR_SUBJECT:
            text = "Waiting for subject"
            color = yellow
        elif gstate == gui_state.ANALYZING:
            text = "Analyzing..."
            color = yellow
        elif gstate == gui_state.HEART_RATE_MONITORING_IN_PROGRESS:
            text = "Heart Rate Monitoring in progress"
            color = green
        else:
             text = "Uknown State"
             color = (255,255,255)
        
        cv2.putText(display_frame, status_txt, 
                    ((frame_offset_x * 2) + live_image.shape[1],
                     logos_y_offset + (2 * st_logo.shape[0])),
                    FONT, fs, color, THICK, AA)
    
    state = gui_state.INITIALIZING
    status_txt = "Initializing..."
    cam = cv2.VideoCapture(1)
    ret, frame = cam.read(0)
    frame = cv2.resize(frame, dsize=(896,1075))
    # Draw the permanent frame with the logos
    # Top banner
    blank_frame[frame_offset_y : frame_offset_y + project_banner.shape[0],
                (2 * frame_offset_x) + frame.shape[1] : (2 * frame_offset_x) + frame.shape[1] + project_banner.shape[1],
                :] = project_banner
    # ST Logo
    logos_y_offset = int(1.5 * frame_offset_y) + project_banner.shape[0]
    logos_x_offset = (frame_offset_x * 2) + frame.shape[1]
    blank_frame[logos_y_offset : logos_y_offset + st_logo.shape[0], 
                logos_x_offset : logos_x_offset + st_logo.shape[1], 
                :] = st_logo
    # Nuralogix Logo
    logos_x_offset += (st_logo.shape[1] + int(1 * frame_offset_x))
    blank_frame[logos_y_offset : logos_y_offset + nuralogix_logo.shape[0], 
                logos_x_offset : logos_x_offset + nuralogix_logo.shape[1], 
                :] = nuralogix_logo
    # Sensor Text
    cv2.putText(blank_frame, "VB56GxA 1.5MP IR Sensor", 
                    ((blank_frame.shape[1] - 500),
                     blank_frame.shape[0]),
                    FONT, fs, (255,255,255), THICK, AA)
    

    while True:
        display_frame = np.copy(blank_frame)
        if hr_event.is_set():
             status_txt = get_hr_message()[0:2] + " BPM"
             print(f"GUI received ->{status_txt}")
        frame = cv2.resize(frame, dsize=(896,1075))
        draw_frame(frame, state)
        cv2.imshow("Result", display_frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            write_shm_message(status_shm, status_ready, end_process_str)
            return 0
        ret, frame = cam.read()
    return 1
    
if __name__ == "__main__":
    ARRAY_SIZE = 512
    # Create shared memory and event flags to 2 processes
    # Status from worker to coordinator
    status_shm_1 = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
    status_shm_2 = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
    status_shm_gui = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
    coordinator_worker_shm_1 = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
    coordinator_worker_shm_2 = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
    # HR data from workers to coordinator
    hr_shm_1 = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
    hr_shm_2 = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
    hr_shm_gui = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)

    start_event_1 = multiprocessing.Event() # This even is sent from coordinator to workers to start HR
    start_event_1.clear()
    start_event_2 = multiprocessing.Event()
    start_event_2.clear()
    hr_ready_event_1 = multiprocessing.Event() # This event is sent from worker to coordinator to read a new HR
    hr_ready_event_1.clear()
    hr_ready_event_2 = multiprocessing.Event()
    hr_ready_event_2.clear()
    hr_ready_event_gui = multiprocessing.Event()
    hr_ready_event_gui.clear()
    status_event_1 = multiprocessing.Event() # This event is sent from worker to coordinator to read a new status
    status_event_1.clear()
    status_event_2 = multiprocessing.Event()
    status_event_2.clear()
    status_event_gui = multiprocessing.Event()
    status_event_gui.clear()

    # Establish worker processes
    set_active_process(1)
    worker_1 = HRM_Proc(status_shm_1, hr_shm_1,coordinator_worker_shm_1,start_event_1,status_event_1,hr_ready_event_1,1)
    worker_2 = HRM_Proc(status_shm_2, hr_shm_2,coordinator_worker_shm_2,start_event_2,status_event_2,hr_ready_event_2,2)
    worker_arr = [worker_1, worker_2]
    hr_coordinator_proc = Process(target=manage_hr_process, name="hr_proc", args=(worker_arr, hr_shm_gui, hr_ready_event_gui,))
    hr_coordinator_proc.start()
    gui_process(hr_shm_gui,hr_ready_event_gui,status_shm_gui,status_event_gui)
