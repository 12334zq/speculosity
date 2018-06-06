import matplotlib.pyplot as plt
from tensorflow.python.keras.callbacks import Callback


class PlotMetrics(Callback):
    def __init__(self, plot_loss=True, plot_acc=False):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.plot_acc = plot_acc
        self.plot_loss = plot_loss
        self.n_ax = int(plot_acc) + int(plot_loss)
        self.hasrun = False

    def on_train_begin(self, logs={}):
        if not self.hasrun:
            self.hasrun = True
            
            # Prepare figure
            if self.n_ax > 0:
                fig = plt.figure()
    
            # Loss plot
            if self.plot_loss:
                self.ax1 = fig.add_subplot(self.n_ax, 1, self.n_ax)
                self.ax1.set_xlabel('Epochs')
                self.ax1.set_ylabel('Loss')
                self.l1 = self.ax1.plot(self.x, self.losses, label="Train loss")[0]
                self.l2 = self.ax1.plot(self.x, self.val_losses, label="Val loss")[0]
                self.ax1.legend()
                plt.grid(True)
    
            # Accuracy plot
            if self.plot_acc:
                self.ax2 = fig.add_subplot(self.n_ax, 1, 1)
                self.ax2.set_ylabel('Accuracy')
                self.l3 = \
                    self.ax2.plot(self.x, self.acc, label="Train accuracy")[0]
                self.l4 = \
                    self.ax2.plot(self.x, self.val_acc, label="Val accuracy")[0]
                self.ax2.legend()
                plt.grid(True)

    def on_epoch_end(self, epoch, logs={}):
        # Loss plot
        self.x.append(self.i)

        if self.plot_loss:
            # Fetch data
            self.losses.append(logs.get('loss'))
            self.val_losses.append(logs.get('val_loss'))

            # Update data
            self.l1.set_xdata(self.x)
            self.l1.set_ydata(self.losses)
            self.l2.set_xdata(self.x)
            self.l2.set_ydata(self.val_losses)
            self.ax1.relim()
            self.ax1.autoscale_view()

        # Accuracy plot
        if self.plot_acc:
            # Fetch data
            self.acc.append(logs.get('acc'))
            self.val_acc.append(logs.get('val_acc'))

            # Update data
            self.l3.set_xdata(self.x)
            self.l3.set_ydata(self.acc)
            self.l4.set_xdata(self.x)
            self.l4.set_ydata(self.val_acc)
            self.ax2.relim()
            self.ax2.autoscale_view()

        # Update plot
        if self.n_ax > 0:
            plt.draw()
            plt.pause(0.01)

        self.i += 1
