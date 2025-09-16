from subprocess import Popen
import os.path
import cv2

start_txt = "start"
process_limit_txt = "process_limit"
measuring_in_progress_txt = "measuring_in_progress"
error_txt = "error"
end_process_txt = "end"
active_app = 1
inactive_app = 2
message_dir = os.path.join(os.path.abspath("."), "messages")
cam = cv2.VideoCapture(1)

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

def write_message(message, app_num):
    f = open(os.path.join(message_dir,message+str(app_num)),"w")
    f.close()


def display_result(app_num):
    ret, frame = cam.read()
    c, r = 2, 15
    r = _draw_text(f"Current Result: {app_num}", frame, (c,r))
    cv2.imshow("Result", frame)
    k = cv2.waitKey(1) & 0xFF
    return k

def clear_messages():
    print("Removing all messages")
    files = os.listdir(message_dir)
    for f in files:
        os.remove(os.path.join(message_dir, f))
    

clear_messages()
p1=Popen("start_demo1.bat")
p2=Popen("start_demo2.bat")
set_active_process(1)
write_message(start_txt, active_app)
exit_flag = False
while not exit_flag:
    while(not(os.path.exists(os.path.join(message_dir, error_txt + str(active_app))))):
        k = display_result(active_app)
        if os.path.exists(os.path.join(message_dir, process_limit_txt+str(active_app))):
            os.remove(os.path.join(message_dir,start_txt+str(active_app)))
            write_message(start_txt, inactive_app)
        if os.path.exists(os.path.join(message_dir, message_dir+str(inactive_app))):
            break
    #############################################
    ########### NEED TO ADD SOME WAIT LOGIC HERE
    #############################################
    set_active_process(inactive_app)
    write_message(end_process_txt, inactive_app)
    #while(not(os.path.exists(os.path.join(message_dir, process_limit_txt+str(active_app)))
    #        or
    #        os.paht.exists(os.path.join(message_dir, error_txt + str(active_app))))):
    #    k = display_result(active_app)
