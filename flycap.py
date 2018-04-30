import PyCapture2
import matplotlib.pyplot as plt
import matplotlib.animation as anim
import numpy as np

# Location of saving image
image_dir = "./data/roughness/1/B8/"

# Camera serial number (find it by opening FlyCap2 Software)
serial_number = 12460852

# Camera properties
frame_rate = 120    # [fps]
autoExpose = True
shutter = 4         # [ms], used only if autoExpose == Flase
gain = 0            # [db], used only if autoExpose == Flase


def printBuildInfo():
    libVer = PyCapture2.getLibraryVersion()
    print("PyCapture2 library version: ",
          libVer[0], libVer[1], libVer[2], libVer[3])
    print()


def printCameraInfo(cam):
    camInfo = cam.getCameraInfo()
    print("\n*** CAMERA INFORMATION ***\n")
    print("Serial number - ", camInfo.serialNumber)
    print("Camera model - ", camInfo.modelName)
    print("Camera vendor - ", camInfo.vendorName)
    print("Sensor - ", camInfo.sensorInfo)
    print("Resolution - ", camInfo.sensorResolution)
    print("Firmware version - ", camInfo.firmwareVersion)
    print("Firmware build time - ", camInfo.firmwareBuildTime)
    print()


def update(i, cam, axis, img_shape):
    global autoExpose

    # Grabe new frame
    img = cam.retrieveBuffer().convert(PyCapture2.PIXEL_FORMAT.RAW8)
    data = img.getData().reshape(img_shape)[0::2, 0::2]
    axis.set_data(data)

    # Grab current camera properties
    shutter = cam.getProperty(PyCapture2.PROPERTY_TYPE.SHUTTER).absValue
    gain = cam.getProperty(PyCapture2.PROPERTY_TYPE.GAIN).absValue

    # Exposition adjustemnt
    max_val = np.max(data)
    print("Shutter = {0:.2f}[ms], Gain = {1:.1f}[db], Max pixel value = {2:d}    ".
          format(shutter, gain, max_val), end="\r")

    if max_val == 255:
        if gain == 0:
            if shutter > 0.1:
                shutter = max(0.1, shutter*0.95)
        else:
            gain = max(0, gain - 0.2)

    elif max_val < 235:
        if shutter < 8:
            shutter = min(8.1, shutter*1.05)
        else:
            gain += 0.2

    # Update camera parameters
    if autoExpose:
        cam.setProperty(type=PyCapture2.PROPERTY_TYPE.SHUTTER,
                        autoManualMode=False, absValue=shutter)
        cam.setProperty(type=PyCapture2.PROPERTY_TYPE.GAIN,
                        autoManualMode=False, absValue=gain)


_i = 0
def key_press_handler(event):
    global _i
    if event.key == 'q':
        plt.close(event.canvas.figure)
        cam.stopCapture()
    elif event.key == ' ':
            image_name = image_dir + "{0:d}.raw".format(_i)
            image = cam.retrieveBuffer().convert(PyCapture2.PIXEL_FORMAT.RAW8)
            image.save(image_name.encode(),
                       PyCapture2.IMAGE_FILE_FORMAT.RAW)
            print("\nImage saved : '", image_name, "'")
            _i += 1


if __name__ == "__main__":
    # Print PyCapture2 Library Information
    printBuildInfo()

    # Ensure sufficient cameras are found
    bus = PyCapture2.BusManager()
    numCams = bus.getNumOfCameras()
    print("Number of cameras detected: ", numCams)
    if not numCams:
        print("Error: Insufficient number of cameras. Exiting...")
        quit()

    # Connecting to camera
    cam = PyCapture2.Camera()

    try:
        uid = bus.getCameraFromSerialNumber(serial_number)
        cam.connect(uid)
        printCameraInfo(cam)
    except PyCapture2.Fc2error:
        print("Error: Failed to connecto camera #", serial_number)
        quit()

    # Setting Properties
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

    # Fetching Image
    cam.startCapture()
    img = cam.retrieveBuffer().convert(PyCapture2.PIXEL_FORMAT.RAW8)
    img_shape = (img.getRows(), img.getCols())
    data = img.getData().reshape(img_shape)[0::2, 0::2]

    # PLotting data
    plt.close('all')
    axis1 = plt.imshow(data, cmap='gray')
    plt.colorbar()
    plt.clim(0, 255)

    # Start animation
    anim.FuncAnimation(plt.gcf(), update, fargs=(cam, axis1, img_shape),
                       interval=20)
    # Register event handler
    plt.gcf().canvas.mpl_connect("key_press_event", key_press_handler)

    plt.show()
