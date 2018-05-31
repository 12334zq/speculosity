import PyCapture2
from camera.info import printCameraInfo, printBuildInfo
from camera import Stream
from camera.utilities import KeyEventHandler
import sys

# Location of saving image
image_dir = "./../data/datasets/roughness1000/raw/B2"

# Camera serial number (find it by opening FlyCap2 Software)
serial_number = 12460852

# Camera properties
frame_rate = 120    # [fps]
autoExpose = True
shutter = 4         # [ms], used only if autoExpose == False
gain = 0            # [db], used only if autoExpose == False

# Print PyCapture2 Library Information
printBuildInfo()

# Connecting to camera
bus = PyCapture2.BusManager()
cam = PyCapture2.Camera()

try:
    uid = bus.getCameraFromSerialNumber(serial_number)
    cam.connect(uid)
    printCameraInfo(cam)
except PyCapture2.Fc2error:
    print("Error: Failed to connect to camera #", serial_number)
    sys.exit()

# Setting Properties
if False:
    cam.setProperty(type=PyCapture2.PROPERTY_TYPE.AUTO_EXPOSURE,
                    autoManualMode=False, onOff=False)
    cam.setProperty(type=PyCapture2.PROPERTY_TYPE.FRAME_RATE,
                    autoManualMode=False, absValue=120)
    cam.setProperty(type=PyCapture2.PROPERTY_TYPE.SHUTTER,
                    autoManualMode=False, absValue=shutter)
    cam.setProperty(type=PyCapture2.PROPERTY_TYPE.GAIN,
                    autoManualMode=False, absValue=gain)

# Setting Configuration
cam.setConfiguration(grabMode=PyCapture2.GRAB_MODE.DROP_FRAMES)

# Stream red pixels
cam.startCapture()
stream = Stream(camera=cam, sampling=(2, 2))
fig = stream.start()

# Attach event handler
handler = KeyEventHandler(cam, save_key=' ', save_dir=image_dir)
handler.connect(fig)
