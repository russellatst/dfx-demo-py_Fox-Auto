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
from dfxutils.renderer import Renderer
import pickle
import matplotlib.pyplot as plt
import argparse
import queue

start_str = "start"
process_limit_str = "process_limit"
measuring_in_progress_str = "measuring_in_progress"
error_str = "error"
end_process_str = "end"
exit_txt = "exited"
initialzied_txt = "initialized"
message_dir = os.path.join(os.path.abspath("."), "messages")
parser = argparse.ArgumentParser()
parser.add_argument(
    "-c",
    "--camera",
    help="Set the camera number. Default is 1", default=1, type=int)
parser.add_argument("-sdi", "--suppress_dfx_images", help="Suppress DFX Images from showing. Default: True", default=True, type=bool)
parser.add_argument("-tts", "--time_to_start", help="The amount of time a face must be present to start the HRM. Default: 1", default=1, type=int)
wait_for_user_time = parser.parse_args().time_to_start
suppress = parser.parse_args().suppress_dfx_images
_cam_num = parser.parse_args().camera # 2 is the IR, 1 is back camera, 0 is front
cam = cv2.VideoCapture(_cam_num)
_polygon_persist_counter_max = 3

ARRAY_SIZE = 512
# Create shared memory and event flags to 2 processes
# Status from worker to coordinator
coordinator_worker_shm_1 = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
coordinator_worker_shm_2 = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
landmark_shm_1 = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
landmark_shm_2 = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
landmark_shm_gui = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
# HR data from workers to coordinator
hr_shm_1 = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
hr_shm_2 = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
hr_shm_gui = shared_memory.SharedMemory(create = True, size=ARRAY_SIZE * np.dtype(np.int32).itemsize)
all_shm = [coordinator_worker_shm_1,coordinator_worker_shm_2,landmark_shm_1,landmark_shm_gui,
            hr_shm_1,hr_shm_2,hr_shm_gui]
# Making Queues instead of shms for states
status_queue_1 = multiprocessing.Queue()
status_queue_2 = multiprocessing.Queue()

start_event_1 = multiprocessing.Event() # This even is sent from coordinator to workers to start HR
start_event_1.clear()
start_event_2 = multiprocessing.Event()
start_event_2.clear()
end_event_1 = multiprocessing.Event()
end_event_1.clear()
end_event_2 = multiprocessing.Event()
end_event_2.clear()
hr_end_event = multiprocessing.Event()
hr_end_event.clear()
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
reset_active_event = multiprocessing.Event()
reset_active_event.clear()
landmark_event_1 = multiprocessing.Event() # This event is sent from worker to coordinator to read a new status
landmark_event_1.clear()
landmark_event_2 = multiprocessing.Event()
landmark_event_2.clear()
landmark_event_gui = multiprocessing.Event()
landmark_event_gui.clear()
join_event_1 = multiprocessing.Event()
join_event_1.clear()
join_event_2= multiprocessing.Event()
join_event_2.clear()


class app_state(enum.Enum):
            INITIALIZING = 0
            STANDBY = 1,
            MEASURING = 2,
            LIMIT_REACHED = 3,
            TERMINATED = 4,
            ERROR = 5

class HRM_Proc():
    def __init__(self, status_queue, hr_shm, coordinator_shm, start_event, status_event, hr_event,app_num,landmark_shm,landmark_event, end_event, reset_hrm_event, suppress=True):
        self.status_queue = status_queue
        self.hr_shm = hr_shm
        self.start_event = start_event
        self.status_event = status_event
        self.hr_event = hr_event
        self.app_num = app_num
        self.proc = None
        self.coordinator_shm = coordinator_shm
        self.current_state = app_state.TERMINATED
        self.landmark_shm = landmark_shm
        self.landmark_event = landmark_event
        self.end_event = end_event
        self.suppress = suppress
        self.reset_timer_en = False
        self.reset_timer = time.time()
        self.is_initialized = False
        self.reset_hrm_event = reset_hrm_event
        self.needs_to_be_joined = False
        self.ready_to_relaunch = False
    
    def run_worker(self,seconds_to_wait_before_starting):
        json_config_file = "config1.json"
        
        if self.app_num ==2:
            json_config_file = "config2.json"
        try:
            print(f"creating task {json_config_file}")
            self.proc = Process(target=dfxdemo.dfxdemo.measurement_loop,name=str(self.app_num), args=(json_config_file,_cam_num,120, 
                                                         self.app_num, self.status_queue, self.hr_shm, 
                                                         self.start_event, self.status_event, self.hr_event, 
                                                         self.coordinator_shm,seconds_to_wait_before_starting,
                                                         self.landmark_shm, self.landmark_event, self.end_event,self.reset_hrm_event, 
                                                         self.suppress))
            self.proc.start()
            self.current_state = app_state.INITIALIZING
        except Exception as e:
            print(f"Error creating task {self.app_num}")
            print(e)
            #self.write_shm_message(self.status_shm, self.status_event, error_str)
            #raise e

    def launch_process(self, seconds_to_wait_before_starting):
        print(f"Starting PROC {self.app_num}")
        if self.proc != None:
            print(f"Task {self.app_num} tried to start but is already running")
            return False
        try:
            self.end_event.clear()
            self.run_worker(seconds_to_wait_before_starting)
            self.start_event.clear()
            print(f"Successful start of Task {self.app_num}")
            return True
        except Exception as e:
            print(f"!!!!! Failure starting Task {self.app_num}")
            raise e
    
    def stop_process(self,permanent_close=False):
        print(f"ENDING Task {self.app_num}")
        try:
            if self.proc != None:
                self.is_initialized = False
                self.write_shm_message(self.coordinator_shm, self.end_event, end_process_str)
                self.proc.terminate() # Tells the process to end
                self.end_event.set() # Tells the loop managing DFX to end
                self.reset_hrm_event.set() # Tells DFX to end
            else:
                print(f"Tried to stop {self.app_num}, but it hasn't started.")
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
        try:
            s = self.status_queue.get_nowait()
            return s
        except queue.Empty:
            return None
    
    def get_shm_message(self, shm):
        s = str(bytes(shm.buf[:]).decode()).replace("\x00", "")
        shm.buf[:] = bytearray(shm.buf.nbytes)
        return s
    
    def reset_hrm(self):
        if self.proc != None:
            self.reset_hrm_event.set()
            self.is_initialized = False

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

def lock_camera_autoexposure():
    message_dir = os.path.join(os.path.abspath("."), "PythonCapture_EntronModule")
    f = open(os.path.join(message_dir,"1"),"w")
    f.close()
    print(f"WROTE: aelock")

def enable_camera_autoexposure():
    message_dir = os.path.join(os.path.abspath("."), "PythonCapture_EntronModule")
    f = open(os.path.join(message_dir,"0"),"w")
    f.close()
    print(f"WROTE: aelock")

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
    cv2.imshow("Automotive Heart Rate Monitoring Demo", frame)
    k = cv2.waitKey(1) & 0xFF
    return k

def write_shm_message(shm, event, text):
    shm.buf[:text.__len__()] = bytearray(text, 'utf-8')
    event.set()

def read_shm_object(shm):
        s = str(bytes(shm.buf[:]).decode()).replace("\x00", "")
        shm.buf[:] = bytearray(shm.buf.nbytes)
        return s

def put_queue_message(q,message):
    try:
        q.put_nowait(message)
        return True
    except Exception as e:
        print(f"Error writing to queue: {e}")
        return False

class coordinator_state(enum.Enum):
            INITIALIZING = 0,
            WAITING_FOR_USER= 1,
            ACTIVE_MEASURING = 2,
            START_INACTIVE = 3,
            SWAP_ACTIVE_PROCS = 4,
            ERROR = 5

class gui_state(enum.Enum):
        INITIALIZING = 0,
        WAITING_FOR_SUBJECT = 1,
        ANALYZING = 2,
        HEART_RATE_MONITORING_IN_PROGRESS = 3,
        ERROR = -1
            
def manage_hr_process(hr_shm_gui, hr_event_gui, landmark_shm_gui, landmark_event_gui, manager_end_event,status_queue_gui,reset_active_event):
    worker_1 = HRM_Proc(status_queue_1, hr_shm_1,coordinator_worker_shm_1,start_event_1,status_event_1,hr_ready_event_1,1,landmark_shm_gui, landmark_event_gui, end_event_1,join_event_1, suppress=suppress)
    worker_2 = HRM_Proc(status_queue_2, hr_shm_2,coordinator_worker_shm_2,start_event_2,status_event_2,hr_ready_event_2,2,landmark_shm_gui, landmark_event_gui, end_event_2,join_event_2, suppress=suppress)
    worker_arr = [worker_1, worker_2]
    active_app = 1
    inactive_app = 2
    enable_camera_autoexposure()
    worker_arr[active_app - 1].launch_process(wait_for_user_time)
    worker_arr[inactive_app - 1].launch_process(wait_for_user_time)
    print("manager starting")
    state = coordinator_state.INITIALIZING
    last_state = coordinator_state.INITIALIZING
    while True:
        #if worker_arr[active_app - 1].status_event.is_set():
        active_message_str = worker_arr[active_app - 1].get_status_message()
        inactive_message_str = worker_arr[inactive_app - 1].get_status_message()
        if active_message_str != None:
            print(f"Active app ({active_app}) message received: {active_message_str}")
            # Initialization finished
            if active_message_str == initialzied_txt and state == coordinator_state.INITIALIZING:
                worker_arr[active_app - 1].start_event.set()
                state = coordinator_state.WAITING_FOR_USER
            # A user is found and measurement has begun
            elif active_message_str == measuring_in_progress_str and state == coordinator_state.WAITING_FOR_USER:
                lock_camera_autoexposure()
                print("Measuring in progress started.")
                state = coordinator_state.ACTIVE_MEASURING
            # The main process has met the threshold limit and the new process should begin
            elif active_message_str == process_limit_str and state == coordinator_state.ACTIVE_MEASURING:
                print(f"Limit received from {active_app}")
                # worker_arr[active_app - 1].current_state = app_state.LIMIT_REACHED
                print("Starting inactive")
                worker_arr[inactive_app - 1].start_event.set()
                worker_arr[active_app - 1].start_event.clear()
                print(f"Starting heart rate for proc: {inactive_app}")
                state = coordinator_state.START_INACTIVE
                print(f"Worker started: {active_app}")
            # Restart the active app
            # If in INACTIVE_STARTED, both active and inactive must be restarted. Coordinator set to Initializing
            # Otherwise, the inactive app becomes the active app and is either not started or in transition
            elif active_message_str == end_process_str or active_message_str.__contains__(error_str):
                print(f"END FROM ACTIVE APP {active_app}")
                if active_message_str.__contains__(error_str):
                    print("Ending due to error:")
                    print(active_message_str)
                    if not active_message_str.__contains__("Failed because no face detected"):
                        worker_arr[active_app - 1].current_state = app_state.ERROR
                        write_shm_message(hr_shm_gui,hr_event_gui,error_str)
                        break
                print(f"Restarting active app: {active_app}")
                enable_camera_autoexposure()
                if state == coordinator_state.START_INACTIVE:
                    worker_arr[active_app - 1].reset_hrm()
                    worker_arr[inactive_app - 1].reset_hrm()
                    state = coordinator_state.INITIALIZING
                # If the inactive has not been start, assume it has initialized and set it to waiting for user
                else:
                    print("switching apps and going into a new state")
                    active_app, inactive_app = set_active_process(inactive_app)
                    worker_arr[inactive_app - 1].start_event.clear()
                    if worker_arr[active_app - 1].is_initialized:
                        state = coordinator_state.WAITING_FOR_USER
                    else:
                        state = coordinator_state.INITIALIZING
                    # handle the now inactive app.
                    worker_arr[inactive_app - 1].reset_hrm()
                    worker_arr[active_app - 1].start_event.set()
            print(f"Active app {active_app} state: {state}")

        # If the limit is reached on the active AND the inactive is ready, swap to the inactive and kill the active
        if state == coordinator_state.START_INACTIVE and worker_arr[inactive_app - 1].hr_event.is_set():
            hr_str = worker_arr[inactive_app - 1].get_hr_message()
            worker_arr[inactive_app - 1].hr_event.clear()
            if not hr_str.__contains__("No"):
                print("Swapping processes")
                worker_arr[active_app - 1].reset_hrm()
                active_app, inactive_app = set_active_process(inactive_app)
                write_shm_message(hr_shm_gui,hr_event_gui,hr_str)
                state = coordinator_state.ACTIVE_MEASURING
                worker_arr[active_app - 1].hr_event.clear()
        # Handling GUI communication
        if worker_arr[active_app - 1].hr_event.is_set():
            hr_str = worker_arr[active_app - 1].get_hr_message()
            if hr_str.__contains__("No"):
                pass
            elif int(hr_str[0]) > 1:
                hr_str = hr_str[0:2] + "." + hr_str[2:]
            else:
                hr_str = hr_str[0:3] + "." + hr_str[3:]
            write_shm_message(hr_shm_gui,hr_event_gui,hr_str)
            print(f"Active HR is {hr_str}")
            worker_arr[active_app - 1].hr_event.clear()

        # Inactive worker handling
        if inactive_message_str != None:
            print(f"Inactive app ({inactive_app}) message received: {inactive_message_str}")
            # Set the flag that the inactive process is ready
            if inactive_message_str == initialzied_txt:
                print(f"Inactive app {inactive_app} is initialized")
                worker_arr[inactive_app - 1].is_initialized = True
            # Handle errors. CURRENT IMPLEMENTATION: the hrm resets itself
            elif inactive_message_str.__contains__(error_str):
                    print(f"ERROR FROM INACTIVE APP {inactive_app}")
                    worker_arr[inactive_app - 1].current_state = app_state.ERROR
            print(f"Inactive app {inactive_app} state: {state}")

        if worker_arr[active_app - 1].landmark_event.is_set():
            landmark_shm_gui.buf[:] = worker_arr[active_app - 1].landmark_shm.buf[:]
            landmark_event_gui.set()
            worker_arr[active_app - 1].landmark_event.clear()
        # Kill the process if this event is set.
        if manager_end_event.is_set():
            print("Manager is closing!")
            for w in worker_arr:
                print(f"Killing process {w.app_num}")
                w.stop_process()
            print("Ending HR Manager")
            for w in worker_arr:
                w.proc.join()
            return
        # Handle face jumping
        # If face jump is found during an active measurement state: end the process and switch or end both processes
        if reset_active_event.is_set():
            reset_active_event.clear()
            print("!!!!Face has jumped!!!!")
            if (state == coordinator_state.ACTIVE_MEASURING):
                print("switching apps and going into a new state")
                worker_arr[active_app - 1].start_event.clear()
                worker_arr[active_app - 1].reset_hrm()
                active_app, inactive_app = set_active_process(inactive_app)
                state = coordinator_state.WAITING_FOR_USER
                worker_arr[active_app - 1].start_event.set()
            elif state == coordinator_state.START_INACTIVE:
                print("Resetting both processes")
                worker_arr[active_app - 1].reset_hrm()
                worker_arr[inactive_app - 1].reset_hrm()
                state = coordinator_state.INITIALIZING

        # Handling GUI states
        if last_state != state:
            print(f"Updating state from {last_state.name} to {state.name}")
            last_state = state
            if state == coordinator_state.INITIALIZING:
                put_queue_message(status_queue_gui,gui_state.INITIALIZING.name)
            elif (state == coordinator_state.ACTIVE_MEASURING):
                put_queue_message(status_queue_gui,gui_state.ANALYZING.name)
            elif state == coordinator_state.START_INACTIVE:
                pass
            elif state == coordinator_state.WAITING_FOR_USER:
                put_queue_message(status_queue_gui,gui_state.WAITING_FOR_SUBJECT.name)
            elif state == coordinator_state.SWAP_ACTIVE_PROCS:
                put_queue_message(status_queue_gui,gui_state.INITIALIZING.name)
            else:
                print("Uknown state. Sending error state")
                put_queue_message(status_queue_gui,gui_state.ERROR.name)

        time.sleep(1)

def gui_process(hr_shm, hr_event, status_queue_gui, reset_active_event,landmark_shm_gui,landmark_event_gui,end_manager_event):
    output_x = 1920
    output_y = 1080
    frame_offset_y = 20
    frame_offset_x = 20
    live_image_dim = (750,900)
    FONT = cv2.FONT_HERSHEY_PLAIN
    AA = cv2.LINE_AA
    THICK = 4
    fs = 3
    fg = (255, 255, 255)
    blank_frame = np.zeros((output_y, output_x, 3), dtype=np.uint8)
    st_logo = cv2.imread(os.path.join(os.path.abspath("."),"assets","ST_logo_2024_black.png"))
    st_logo = cv2.resize(st_logo, (0,0), fx= 0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    project_banner = cv2.imread(os.path.join(os.path.abspath("."),"assets","Project-Banner.png"))
    #project_banner = cv2.resize(project_banner, (0,0), fx=0.5, fy=0.5)
    nuralogix_logo = cv2.imread(os.path.join(os.path.abspath("."),"assets","nura_black.png"))
    nuralogix_logo = cv2.resize(nuralogix_logo, (0,0), fx= 0.7, fy=0.7, interpolation=cv2.INTER_AREA)
    heart_icon = cv2.imread(os.path.join(os.path.abspath("."),"assets","heart.png"))
    heart_icon = cv2.resize(heart_icon, dsize =(120,100))
    sensor_image = cv2.imread(os.path.join(os.path.abspath("."),"assets","sensor_img.png"))
    #sensor_image = cv2.resize(sensor_image, (0,0), fx= 0.3, fy=0.3, interpolation=cv2.INTER_AREA)
    multiplier = 1.5
    input_size = [341, 282]
    hr_history_arr = [0] * 10
    hr_fig = plt.figure()
    plt.ylabel("Heart Rate (BPM)")
    plt.xlabel("Measurment Number")
    last_polygons = None
    polygon_persist_counter = _polygon_persist_counter_max
    excursion_pct = 0.025

    def get_hr_message():
        s = str(bytes(hr_shm.buf[:]).decode()).replace("\x00", "")
        #print(f"FROM IN GUI: {s}")
        hr_shm.buf[:] = bytearray(hr_shm.buf.nbytes)
        hr_event.clear()
        return s
    
    def get_queue_message(q):
        # s = str(bytes(self.status_shm.buf[:]).decode()).replace("\x00", "")
        # self.status_shm.buf[:] = bytearray(self.status_shm.buf.nbytes)
        try:
            s = q.get_nowait()
            return s
        except queue.Empty:
            return None
    
    def draw_polygons_mask(polygons, image):
        #image = cv2.flip(image, 1)
        for polygon in polygons:
            cv2.polylines(image, 
                            [np.round(np.array(polygon) * multiplier).astype(int)],
                        isClosed=True,
                        color=(255, 255, 0),
                        thickness=1,
                        lineType=cv2.LINE_AA)
        #image = cv2.flip(image, 1)
        return image
    
    def draw_frame(live_image,gstate,hr_txt, last_polygons, polygon_persist_counter):
        
        polygons = pickle.loads(landmark_shm_gui.buf[:]) if landmark_event_gui.is_set() else None
        face_found = False
        face_jumped = False
        
        live_image = cv2.flip(live_image, 1)
        if polygons is not None and len(polygons) > 1:
            live_image = draw_polygons_mask(polygons, live_image)
            landmark_event_gui.clear()
            if last_polygons is not None:
            # Check if the face has moved too far
                all_points = np.array([point for polygon in polygons for point in polygon])
                current_all_points_avg = (np.mean(all_points[:,0]),np.mean(all_points[:,1]))
                all_points = np.array([point for polygon in last_polygons for point in polygon])
                last_all_points_avg = (np.mean(all_points[:,0]),np.mean(all_points[:,1]))
                if np.absolute(last_all_points_avg[0] - current_all_points_avg[0] + last_all_points_avg[1] - current_all_points_avg[1]) > 50:
                    face_jumped = True
                    print(f"Last center: {last_all_points_avg}")
                    print(f"New center: {current_all_points_avg}")
                else:
                    last_all_points_avg = current_all_points_avg
            last_polygons = polygons
            face_found = True
            polygon_persist_counter = _polygon_persist_counter_max
        elif last_polygons is not None and len(last_polygons) > 1 and polygon_persist_counter > 0:
            live_image = draw_polygons_mask(last_polygons, live_image)
            face_found = True
            polygon_persist_counter -= 1
        else:
            last_polygons = None

        green = (0, 255, 0)
        yellow = (104, 232, 2444)
        white = (255, 255, 255)
        red = (0, 0, 255)
        display_frame[frame_offset_y : frame_offset_y + live_image.shape[0], 
                       frame_offset_x : frame_offset_x + live_image.shape[1],
                       :] = live_image
        no_hr_txt = "   BPM"
        if gstate == gui_state.INITIALIZING:
            status_txt = "Initializing"
            hr_txt = no_hr_txt
            color = red
        elif gstate == gui_state.WAITING_FOR_SUBJECT:
            status_txt = "Waiting for subject"
            hr_txt = no_hr_txt
            if face_found:
                status_txt = "User Found! Analysis will begin shortly"
            color = yellow
        elif gstate == gui_state.ANALYZING:
            status_txt = "User found! Please wait ~15 seconds"
            hr_txt = "Analysis in progress."
            color = yellow
        elif gstate == gui_state.HEART_RATE_MONITORING_IN_PROGRESS:
            status_txt = "Heart Rate Monitoring in progress"
            color = green
        elif gstate == gui_state.ERROR:
            status_txt = "ERROR. Please restart."
            hr_txt = no_hr_txt
            color = red
        else:
             status_txt = "Uknown State"
             color = (255,255,255)
        
        # Draw the Status text
        cv2.putText(display_frame, status_txt, 
                    status_text_coord,
                    FONT, fs, color, THICK, AA)
        # Draw the HR text
        cv2.putText(display_frame, hr_txt, 
                    bpm_text_coord,
                    FONT, 5, white, THICK, AA)

        return last_polygons, polygon_persist_counter, face_jumped
        
    
    state = gui_state.INITIALIZING
    status_txt = "Initializing..."
    hr_txt = "0 BPM"
    ret, frame = cam.read(0)
    frame = cv2.resize(frame, dsize=live_image_dim)
    multiplier = 0.5 * np.mean([(frame.shape[0] / input_size[0]), (frame.shape[1] / input_size[1])])
    # Draw the permanent frame with the logos
    logos_y_offset = frame_offset_y
    logos_x_offset = int(np.mean(((2 * frame_offset_x) + frame.shape[1], output_x))) - int(project_banner.shape[1] / 2)
    # Top banner
    blank_frame[frame_offset_y : frame_offset_y + project_banner.shape[0],
                logos_x_offset : logos_x_offset + project_banner.shape[1],
                :] = project_banner
    # ST Logo
    st_y_offset = int(2 * frame_offset_y) + project_banner.shape[0]
    st_x_offset = logos_x_offset + int((st_logo.shape[1] + frame_offset_x + nuralogix_logo.shape[1]) / 3)
    blank_frame[st_y_offset : st_y_offset + st_logo.shape[0], 
                st_x_offset : st_x_offset + st_logo.shape[1], 
                :] = st_logo
    # Heart Icon
    heart_icon_y_offset = st_y_offset + (3 * frame_offset_y) + st_logo.shape[0]
    blank_frame[heart_icon_y_offset : heart_icon_y_offset + heart_icon.shape[0], 
                logos_x_offset : logos_x_offset + heart_icon.shape[1], 
                :] = heart_icon
    bpm_text_coord = (logos_x_offset + heart_icon.shape[1] + frame_offset_x, 
                      heart_icon_y_offset + heart_icon.shape[0])
    # Nuralogix Logo
    nura_y_offset = st_y_offset + int(st_logo.shape[0] / 2) - int(nuralogix_logo.shape[0] / 2)
    nura_x_offset = (st_x_offset + frame_offset_x + st_logo.shape[1])
    blank_frame[nura_y_offset : nura_y_offset + nuralogix_logo.shape[0], 
                nura_x_offset : nura_x_offset + nuralogix_logo.shape[1], 
                :] = nuralogix_logo
    # Status Text
    status_text_coord = (logos_x_offset, heart_icon_y_offset + heart_icon.shape[1] + (3 * frame_offset_y))
    # Sensor Text and image
    sensor_text_coord = ((blank_frame.shape[1] - 700),
                     blank_frame.shape[0] - 80)
    cv2.putText(blank_frame, "VB56GxA 1.5MP IR Sensor", 
                    sensor_text_coord,
                    FONT, fs, (255,255,255), THICK, AA)
    # Exit Aplication information
    cv2.putText(blank_frame, "Press 'q' to exit demo", 
                    (10, sensor_text_coord[1]),
                    FONT, fs, (50,50,50), THICK, AA)
    sensor_img_x_offset = int(np.mean((sensor_text_coord[0], blank_frame.shape[1])) - (sensor_image.shape[1] / 2))
    sensor_img_y_offset = sensor_text_coord[1] - sensor_image.shape[0] - frame_offset_y - 30
    blank_frame[sensor_img_y_offset : sensor_img_y_offset + sensor_image.shape[0], 
                sensor_img_x_offset : sensor_img_x_offset + sensor_image.shape[1], 
                :] = sensor_image

    while True:
        display_frame = np.copy(blank_frame)
        #hr_txt = "0 BPM"
        s = get_queue_message(status_queue_gui)
        if s != None:
            print(f"GUI read: {s}")
            #status_ready.clear()
            #s = read_shm_object(status_shm)
            for stat in gui_state:
                if s == stat.name:
                    if state != stat:
                        state = stat
                        print(f"New gui state -> {state.name}")
                    break
        if hr_event.is_set() and ((state == gui_state.ANALYZING) or (state == gui_state.HEART_RATE_MONITORING_IN_PROGRESS)):
            hr_txt = get_hr_message()[0:2]
            if hr_txt.__contains__("No"):
                state = gui_state.ANALYZING
            else: 
                state = gui_state.HEART_RATE_MONITORING_IN_PROGRESS
                hr_txt = str(int(np.round(int(hr_txt) * (1 - excursion_pct)))) + " - " + str(int(np.round(int(hr_txt) * (1 + excursion_pct))))
            #np.roll(hr_history_arr, -1)
            #hr_history_arr[:-1] = int(hr_txt)
            print(f"New GUI BPM set -> {hr_txt}")
            # Draw the graph
            # plt.plot(np.arange(len(hr_history_arr)), hr_history_arr)
            # hr_fig.canvas.draw()
            # plot = np.fromstring(hr_fig.canvas.(), dtype=np.uint8,sep='')
            # plot = plot.reshape(hr_fig.canvas.get_width_height()[::-1] + (3,))
            # plot = cv2.cvtColor(plot, cv2.COLOR_RGB2BGR)
            # display_frame[status_text_coord[0] : status_text_coord[0] + plot.shape[0],
            #             status_text_coord[1] : status_text_coord[1] + plot.shape[1],:] = plot
            
        frame = cv2.resize(frame, dsize=live_image_dim)
        last_polygons, polygon_persist_counter, face_jumped = draw_frame(frame, state, hr_txt, last_polygons, polygon_persist_counter)
        if face_jumped and ((state == gui_state.ANALYZING) or (state == gui_state.HEART_RATE_MONITORING_IN_PROGRESS)):
            reset_active_event.set()
        cv2.imshow("Result", display_frame)
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            end_manager_event.set()
            cam.release()
            print("ending gui")
            return 0
        ret, frame = cam.read()
    return 1

if __name__ == "__main__":
    # Making Queues instead of shms for states
    status_queue_gui = multiprocessing.Queue()
    # Establish worker processes
    set_active_process(1)
    hr_coordinator_proc = Process(target=manage_hr_process, name="hr_proc", args=(hr_shm_gui, hr_ready_event_gui,landmark_shm_gui, landmark_event_gui,hr_end_event,status_queue_gui,reset_active_event))
    hr_coordinator_proc.start()
    gui_process(hr_shm_gui,hr_ready_event_gui,status_queue_gui,reset_active_event,landmark_shm_gui,landmark_event_gui,hr_end_event)
    # gui_proc = Process(target=gui_process, name="gui_proc",args=(hr_shm_gui,hr_ready_event_gui,status_shm_gui,status_event_gui,landmark_shm_gui,landmark_event_gui,hr_end_event))
    # gui_proc.start()
    # asyncio.run(manage_hr_process(hr_shm_gui, hr_ready_event_gui,landmark_shm_gui, landmark_event_gui,hr_end_event,status_shm_gui,status_event_gui))
    print("-------ENDING DEMO---------")
    #hr_end_event.set()
    hr_coordinator_proc.join()
    hr_coordinator_proc.close()
    #gui_proc.kill()
    #gui_proc.join()
    for shm in all_shm:
        shm.close()
        shm.unlink()
    exit
    
