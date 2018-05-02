import PyCapture2
import matplotlib.pyplot as plt
import matplotlib.animation as anim


class Stream:
    def __init__(self,
                 camera,
                 frame_rate=20,
                 pixel_format=PyCapture2.PIXEL_FORMAT.RAW8,
                 sampling=(1, 1),
                 sampling_offset=(0, 0),
                 cmap='gray'):
        self.camera = camera
        self.pixel_format = pixel_format
        self.frame_rate = frame_rate
        self.sampling = sampling
        self.sampling_offset = sampling_offset
        self.axis_image = None
        self.cmap = cmap
        self.stop = False

    def _frames(self):
        while ~self.stop:
            # Retrieving image
            image = self.camera.retrieveBuffer()
            image = image.convert(self.pixel_format)
            shape = (image.getRows(), image.getCols())
            yield image.getData().reshape(shape)

    def _update(self, data):
        # Subsampling image
        data = data[self.sampling_offset[0]::self.sampling[0],
                    self.sampling_offset[1]::self.sampling[1]]

        # Updating plot
        self.axis_image.set_data(data)

    def start(self):
        # Creating axis
        fig = plt.figure()
        axis = fig.add_subplot(1, 1, 1)
        self.axis_image = axis.imshow([], cmap=self.cmap)
        plt.colorbar()
        plt.clim(0, 255)

        # Time between frames in ms
        interval = int(1000/self.frame_rate)

        # Start animation
        anim.FuncAnimation(fig=fig,
                           func=self._update,
                           frames=self._frames,
                           interval=interval)

        return fig

    def add_handler(self, event_name, event_handler, args):
        self.fig.canvas.mpl_connect(event_name, event_handler, )
