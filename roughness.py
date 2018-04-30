from keras import models, callbacks
import os
import matplotlib.pyplot as plt
root = os.getcwd()
os.chdir(root + '/lib')
from vgg16 import VGG16
os.chdir(root + '/../datasets')
from data import load

# Setting up plot
plt.close('all')
plt.ion()
fig = plt.figure()
axis1 = fig.add_subplot(211)
line1, = axis1.plot([1, 2, 3])
plt.ylabel("Loss")

axis2 = fig.add_subplot(212)
line2, = axis2.plot([1, 2, 3])
plt.ylabel("Accuracy")
plt.xlabel("batches")

plt.show()


class LossHistory(callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.loss = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        line1.set_data(range(len(self.loss)), self.loss)
#        line2.set_data(range(len(self.acc)), self.acc)
        axis1.set_xlim(0, len(self.loss))
        axis1.set_ylim(0, max(self.loss))
#        axis2.set_xlim(0, len(self.acc))
#        axis2.set_ylim(0, max(self.acc))
        plt.draw()
        plt.pause(0.05)


# Importing data
data = load("roughness", True)
input_shape = (128, 128, 1)
nb_classes = 13

# Parameters
batch_size = 4
filename_load = root + "/models/roughness"
filename_save = filename_load
load = True
save = True

# Load/Creaate Model
if load:
    try:
        model = models.load_model(filename_load, compile=False)
    except OSError:
        print("ERROR : Saved model not found! Exiting!")
        os._exit(0)
else:
    model = VGG16(input_shape, nb_classes, False)

# Train Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x=data.train.data,
          y=data.train.labels,
          batch_size=batch_size,
          epochs=2,
          validation_split=0.1,
          shuffle=True,
          callbacks=[LossHistory()])

# Save model
if save:
    try:
        model.save(filename_save)
    except OSError:
        print("ERROR : An error occured while saving the model. All progress"
              " is lost! ")

# Test network
score = model.evaluate(data.test.data, data.test.labels, batch_size=batch_size)
print("Test loss: {0:2f}, Test Accuracy: {1:2f}".format(score[0], score[1]))
