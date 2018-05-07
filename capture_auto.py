import PyCapture2
from PyAPT import APTMotor
from camera.info import printCameraInfo, printBuildInfo
from camera import Stream
from camera.utilities import KeyEventHandler
import sys
import numpy as np
import time

# Location of saving image
image_dir = "./../data/datasets/roughness1000/raw/B2"

# Camera serial number (find it by opening FlyCap2 Software)
serial_camera = 12460852

# Stage parameters
TDC001 = 31
scan_x = [1, 11]    # [mm]
scan_z = [1, 11]    # [mm]
scan_steps_x = 100
scan_steps_z = 100
serial_z = 83858553
serial_x = 83858439

# Print PyCapture2 Library Information
printBuildInfo()

# Connecting to camera
bus = PyCapture2.BusManager()
cam = PyCapture2.Camera()

try:
    uid = bus.getCameraFromSerialNumber(serial_camera)
    cam.connect(uid)
    printCameraInfo(cam)
except PyCapture2.Fc2error:
    print("Error: Failed to connect to camera #", serial_camera)
    sys.exit()

# Connecting to stage
try:
    z_axis = APTMotor(serial_z, HWTYPE=TDC001)
except:
    print("Error: Failed to connec to Z axis #", serial_z)
    sys.exit()

try:
    x_axis = APTMotor(serial_x, HWTYPE=TDC001)
except:
    print("Error: Failed to connec to X axis #", serial_x)
    sys.exit()

# Homing stage
print("Homing axis Z...\t", end='')
z_axis.go_home()
print("[DONE]\nHoming axis X...\t", end='')
x_axis.go_home()
print("[DONE]")

# Checking scan range
print("Checking scan range...\t", end='')
z_axis.mAbs(scan_z[0]), x_axis.mAbs(scan_x[0])
z_axis.mAbs(scan_z[1]), x_axis.mAbs(scan_x[0])
z_axis.mAbs(scan_z[1]), x_axis.mAbs(scan_x[1])
z_axis.mAbs(scan_z[0]), x_axis.mAbs(scan_x[1])
z_axis.mAbs(scan_z[0]), x_axis.mAbs(scan_x[0])
print("[DONE]")

# Setting Configuration
cam.setConfiguration(grabMode=PyCapture2.GRAB_MODE.DROP_FRAMES)

# Stream red pixels
cam.startCapture()
stream = Stream(camera=cam, sampling=(2, 2))
fig = stream.start()

# Attach event handler
handler = KeyEventHandler(cam)
handler.connect(fig)

# Begin scan
i = 0
for z in np.linspace(scan_z[0], scan_z[1], scan_steps_z):
    z_axis.mAbs(z)
    for x in np.linspace(scan_x[0], scan_x[1], scan_steps_x):
        print("Moving to ({0:.3f},{1:.3f})".format(x,z))
        x_axis.mAbs(x)
        time.sleep(0.5)
        image_name = image_dir + "/{0:d}.raw".format(i)
        image = cam.retrieveBuffer()
        image.save(image_name.encode(), PyCapture2.IMAGE_FILE_FORMAT.RAW)

        print("Image saved : '", image_name, "'\n")
        i += 1
        
# Scan terminated
print("Scan terminated")

# Close peripherals
z_axis.cleanUpAPT()
x_axis.cleanUpAPT()