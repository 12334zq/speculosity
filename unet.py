from tensorflow.python.keras import layers, models, Input, optimizers, callbacks
import imageio
import numpy as np
import random
import matplotlib.pyplot as plt

# Model parameters
image_shape = (None, None, 1)   # Monochrome image
image_size = 256
train_size = 1000
val_size = 50
test_size = 100
n_epochs = 10
data_set_loc = "../../data/datasets/bpm/rings"
model_loc = "./model_fwgn"

# Importing Data
train_images = list()
train_labels = list()
val_images = list()
val_labels = list()
test_images = list()
test_labels = list()

for i in range(train_size):
    train_images.append(
        imageio.imread(data_set_loc + "/train/images/{0:d}.bmp".format(i+1))
    )
    train_labels.append(
        imageio.imread(data_set_loc + "/train/labels/{0:d}.bmp".format(i+1))
    )

for i in range(val_size):
    val_images.append(
        imageio.imread(data_set_loc + "/val/images/{0:d}.bmp".format(i+1))
    )
    val_labels.append(
        imageio.imread(data_set_loc + "/val/labels/{0:d}.bmp".format(i+1))
    )

for i in range(test_size):
    test_images.append(
        imageio.imread(data_set_loc + "/test/images/{0:d}.bmp".format(i+1))
    )
    test_labels.append(
        imageio.imread(data_set_loc + "/test/labels/{0:d}.bmp".format(i+1))
    )

train_images = np.array(train_images).reshape(-1, image_size, image_size, 1)
train_labels = np.array(train_labels).reshape(-1, image_size, image_size, 1)
val_images = np.array(val_images).reshape(-1, image_size, image_size, 1)
val_labels = np.array(val_labels).reshape(-1, image_size, image_size, 1)
test_images = np.array(test_images).reshape(-1, image_size, image_size, 1)
test_labels = np.array(test_labels).reshape(-1, image_size, image_size, 1)


# U-net model
nb_layers = 32   # Number of filters in the first unet layer

def create_model():
    inputs = Input(shape=image_shape)

    conv1 = layers.Conv2D(nb_layers, (3, 3), activation='relu', padding='same')(inputs)
    conv1 = layers.Conv2D(nb_layers, (3, 3), activation='relu', padding='same')(conv1)
    pool1 = layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = layers.Conv2D(nb_layers*2, (3, 3), activation='relu', padding='same')(pool1)
    conv2 = layers.Conv2D(nb_layers*2, (3, 3), activation='relu', padding='same')(conv2)
    pool2 = layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = layers.Conv2D(nb_layers*4, (3, 3), activation='relu', padding='same')(pool2)
    conv3 = layers.Conv2D(nb_layers*4, (3, 3), activation='relu', padding='same')(conv3)
    pool3 = layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = layers.Conv2D(nb_layers*8, (3, 3), activation='relu', padding='same')(pool3)
    conv4 = layers.Conv2D(nb_layers*8, (3, 3), activation='relu', padding='same')(conv4)
    pool4 = layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = layers.Conv2D(nb_layers*16, (3, 3), activation='relu', padding='same')(pool4)
    conv5 = layers.Conv2D(nb_layers*16, (3, 3), activation='relu', padding='same')(conv5)

    up_conv5 = layers.UpSampling2D(size=(2, 2))(conv5)
    up6 = layers.concatenate([up_conv5, conv4], axis=3)
    conv6 = layers.Conv2D(nb_layers*8, (3, 3), activation='relu', padding='same')(up6)
    conv6 = layers.Conv2D(nb_layers*8, (3, 3), activation='relu', padding='same')(conv6)

    up_conv6 = layers.UpSampling2D(size=(2, 2))(conv6)
    up7 = layers.concatenate([up_conv6, conv3], axis=3)
    conv7 = layers.Conv2D(nb_layers*4, (3, 3), activation='relu', padding='same')(up7)
    conv7 = layers.Conv2D(nb_layers*4, (3, 3), activation='relu', padding='same')(conv7)

    up_conv7 = layers.UpSampling2D(size=(2, 2))(conv7)
    up8 = layers.concatenate([up_conv7, conv2], axis=3)
    conv8 = layers.Conv2D(nb_layers*2, (3, 3), activation='relu', padding='same')(up8)
    conv8 = layers.Conv2D(nb_layers*2, (3, 3), activation='relu', padding='same')(conv8)

    up_conv8 = layers.UpSampling2D(size=(2, 2))(conv8)
    up9 = layers.concatenate([up_conv8, conv1], axis=3)
    conv9 = layers.Conv2D(nb_layers, (3, 3), activation='relu', padding='same')(up9)
    conv9 = layers.Conv2D(nb_layers, (3, 3), activation='relu', padding='same')(conv9)

    conv10 = layers.Conv2D(1, (1, 1))(conv9)

    return models.Model(inputs=inputs, outputs=conv10)

# Training network
try:
    model = models.load_model(model_loc, compile=False)
except OSError:
    print("Saved model not found, creating new model.")
    model = create_model()

model.compile(optimizer=optimizers.Adam(),
              loss="mse",
              metrics=['accuracy'])

if False:
    model.fit(train_images, 
              train_labels, 
              validation_data=(val_images, val_labels),
              batch_size=32, 
              epochs=50,
              callbacks=[callbacks.TensorBoard()])

print("Saving current model.")
model.save(model_loc)

# Test network
score = model.evaluate(test_images, test_labels, batch_size=16)
print("Test loss: {0:2f}, Test Accuracy: {1:2f}".format(score[0], score[1]))

# Visualize results
plt.figure(1)

for i in range(5):
    index = random.randint(0, test_size - 1)
    pred = model.predict(test_images[index:index+1, :, :, :])

    # Plot input image
    ax = plt.subplot(3, 5, i+1)
    plt.imshow(test_images[index, :, :, 0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Input {0:d}".format(index))

    # Plot label
    ax = plt.subplot(3, 5, i+6)
    plt.imshow(test_labels[index, :, :, 0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("True {0:d}".format(index))

    # Plot prediction
    ax = plt.subplot(3, 5, i+11)
    plt.imshow(pred[0, :, :, 0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Predicted {0:d}".format(index))

plt.show()
