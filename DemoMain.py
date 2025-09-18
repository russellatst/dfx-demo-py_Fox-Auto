import os.path
import cv2
import dfxdemo.dfxdemo
import multiprocessing
from multiprocessing import Process, shared_memory
import numpy as np
import asyncio
import time

start_str = "start"
process_limit_str = "process_limit"
measuring_in_progress_str = "measuring_in_progress"
error_str = "error"
end_process_str = "end"
message_dir = os.path.join(os.path.abspath("."), "messages")
cam = cv2.VideoCapture(1)

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
    
    def run_worker(self):
        json_config_file = "config1.json"
        if self.app_num ==2:
            json_config_file = "config2.json"
        try:
            asyncio.run(dfxdemo.dfxdemo.run_measurements(json_config_file,2,120, self.app_num, self.status_shm, self.hr_shm, self.start_event, self.status_event, self.hr_event, self.coordinator_shm))
        except Exception as e:
            raise e

    def start_process(self):
        self.proc = Process(target=self.run_worker, name=str(self.app_num))
        self.proc.start()
        return self.proc
    
    def stop_process(self):
        print(f"ENDING PROC {self.app_num}")
        self.write_shm_message(self.coordinator_shm, self.start_event,end_process_str)
        self.start_event.set()
        if self.proc != None:
            self.proc.terminate()
            self.proc = None

    def write_shm_message(shm, event, text):
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

def manage_hr_process(worker_arr):
    active_app = 1
    inactive_app = 2
    active_proc = worker_arr[active_app - 1].start_process()
    while True:
        print(f"Worker started: {active_app}")
        worker_arr[active_app - 1].hr_event.wait()
        worker_arr[active_app - 1].hr_event.clear()
        worker_arr[active_app - 1].start_event.set()
        print("Start Sent")
        # TODO: Relay HRM to UI
        # Wait for a status
        while True:
            worker_arr[active_app - 1].status_event.wait()
            worker_arr[active_app - 1].status_event.clear()
            print("DEBUG: Status event received!")
            message_str = worker_arr[active_app - 1].get_status_message()
            print(f"{message_str}")
            if message_str == process_limit_str:
                active_proc = worker_arr[inactive_app - 1].start_process()
                worker_arr[inactive_app - 1].hr_event.wait()
                worker_arr[inactive_app - 1].hr_event.clear()
                # If process limit reached, start the next one
                worker_arr[inactive_app - 1].start_event.set()
                print(f"Started worker: {inactive_app}")
                worker_arr[inactive_app - 1].status_event.wait()
                break
            elif message_str == error_str:
                print("Oh no! Error! Exiting.")
                time.sleep(4)
                return 0
        # Wait for the new HR to get going.
        worker_arr[active_app - 1].stop_process()
        active_app, inactive_app = set_active_process(inactive_app)
        #worker_arr[inactive_app - 1] = HRM_Proc(worker_arr[inactive_app - 1].status_shm, worker_arr[inactive_app - 1].hr_shm, worker_arr[inactive_app - 1].start_event, worker_arr[inactive_app - 1].status_event, worker_arr[inactive_app - 1].hr_event, inactive_app)

def gui_process(hr_shm, hr_ready, status_shm, status_ready):
    print("This is where the GUI goes.")
    return
    
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
    # Clear buffers and create arrays to interact with the data. Personal preference, but they're easier to deal with than shm directly
    # status_arr_1 = np.ndarray((ARRAY_SIZE,), dtype=np.int32, buffer=status_shm_1.buf)
    # status_arr_1[:] = 0
    # status_arr_2 = np.ndarray((ARRAY_SIZE,), dtype=np.int32, buffer=status_shm_2.buf)
    # status_arr_2[:] = 0
    # status_arr_gui = np.ndarray((ARRAY_SIZE,), dtype=np.int32, buffer=status_shm_gui.buf)
    # status_arr_gui[:] = 0
    # hr_arr_1 = np.ndarray((ARRAY_SIZE,), dtype=np.int32, buffer=hr_shm_1.buf)
    # hr_arr_1[:] = 0
    # hr_arr_2 = np.ndarray((ARRAY_SIZE,), dtype=np.int32, buffer=hr_shm_2.buf)
    # hr_arr_2[:] = 0
    # hr_arr_gui = np.ndarray((ARRAY_SIZE,), dtype=np.int32, buffer=hr_shm_gui.buf)
    # hr_arr_gui[:] = 0

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
    hr_coordinator_proc = Process(target=manage_hr_process, name="hr_proc", args=(worker_arr,))
    hr_coordinator_proc.start()
    # LOGIC TO HANDLE ERRORS UNEXPECTED QUITS
    #while not error or not quitted:
    #    time.sleep(1)
    # INSERT GRACEFUL SHUT DOWN
