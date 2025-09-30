import cv2


def list_ports():
    for dev_port in range(0,3):
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port,h,w))
            # working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port,h,w))
                #available_ports.append(dev_port)

def show_tested_camera(dev_port):
    camera = cv2.VideoCapture(dev_port)
    while True:
        ret, img = camera.read()
        cv2.imshow(f"test {dev_port}", img)
        k = cv2.waitKey(0)
        if k == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    for dev_port in range(0,3):
        show_tested_camera(dev_port)