import numpy as np
import PyCapture2
import matplotlib.pyplot as plt

_dGain = 0.2
_dShutter = 0.05
_minShutter = 0.1
_maxShutter = 0.8


class KeyEventHandler:
    def __init__(self,
                 camera,
                 exit_key='q',
                 save_key=None,
                 save_dir=None,
                 save_pixel_format=PyCapture2.PIXEL_FORMAT.RAW8):
        """
        Key event handler for matplotlib figure that enables exiting

        :param PyCapture2.Camera() camera:
        :param bytes exit_key:
        :param bytes save_key:
        :param str save_dir:
        :param PyCapture2.PIXEL_FORMAT save_pixel_format:
        """
        self.i = 0
        self.exit_key = exit_key
        self.save_key = save_key
        self.save_dir = save_dir
        self.camera = camera
        self.save_pixel_format = save_pixel_format

        # If save
        if save_key is not None and save_dir is None:
            raise ValueError(
                "If save key is specified, a save directory must also be!")

    def __call__(self, event):
        if event.key == self.exit_key:
            plt.close(event.canvas.figure)
            self.camera.stopCapture()
        elif event.key == self.save_key:
            image_name = self.save_dir + "{0:d}.raw".format(self.i)
            image = self.camera.retrieveBuffer()
            image.save(image_name.encode(), self.save_pixel_format)

            print("\nImage saved : '", image_name, "'")
            self.i += 1

    def connect(self, fig):
        fig.canvas.mpl_connect("key_press_event", self)


def autoExpose(camera,
               target_level=245,
               adjust_shutter=True,
               adjust_gain=True):
    """
    Auto adjust the shutter speed and/or the gain to avoid pixel saturation.

    :param PyCapture2.Camera() camera:
    :param int target_level:
    :param bool adjust_shutter:
    :param bool adjust_gain:
    """

    if target_level <= 0 or target_level >= 255:
        raise ValueError("Target level must be value between in the range"
                         "]0,255[ !")

    # There must be something to adjust
    if ~adjust_shutter and ~adjust_gain:
        raise ValueError("At one of the variables must be adjustable!")

    while True:
        # Grab frame
        image = camera.retrieveBuffer()
        image = image.convert(PyCapture2.PIXEL_FORMAT.RAW8)
        data = image.getData()

        # Grab current camera properties
        shutter = camera.getProperty(PyCapture2.PROPERTY_TYPE.SHUTTER).absValue
        gain = camera.getProperty(PyCapture2.PROPERTY_TYPE.GAIN).absValue

        # Exposition adjustment
        max_val = np.max(data)
        print("Shutter = {0:.2f}[ms], Gain = {1:.1f}[db],"
              "Max pixel value = {2:d}  ".format(shutter, gain, max_val),
              end='\r')

        if max_val == max:
            if gain == 0 or ~adjust_shutter:
                if shutter > 0.1:
                    shutter = max(0.1, shutter * (1 + _dShutter))
            else:
                gain = max(0, gain - _dGain)

        elif max_val < min:
            if shutter < 8:
                shutter = min(8.1, shutter / (1 + _dShutter))
            else:
                gain += _dGain
        else:
            break

        # Update camera parameters
        if autoExpose:
            camera.setProperty(type=PyCapture2.PROPERTY_TYPE.SHUTTER,
                               autoManualMode=False, absValue=shutter)
            camera.setProperty(type=PyCapture2.PROPERTY_TYPE.GAIN,
                               autoManualMode=False, absValue=gain)
