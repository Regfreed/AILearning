'''
Izgradnja klasifikatora prometnih znakova pomocu Keras API koristenjem ImageGeneratora i augmentacije.
'''

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from matplotlib import pyplot as plt
import numpy as np


batch_size = 32
train_datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.1,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)


train_generator = train_datagen.flow_from_directory(
    directory="gtsrb_dataset/Train_divide/train",
    target_size=(48, 48),
    color_mode="rgb",
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=True,
    seed=42)

validation_generator = test_datagen.flow_from_directory(
        directory="gtsrb_dataset/Train_divide/val",
        target_size=(48, 48),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle = False)

test_generator = test_datagen.flow_from_directory(
        directory="gtsrb_dataset/Test",
        target_size=(48, 48),
        batch_size=1,
        class_mode='categorical',
        shuffle=False)



# plot one batch of maximally 32 images
batch = train_generator.next()
plt.subplots(figsize=(10, 8))
for i in range(1,batch_size+1):
	plt.subplot(4,8,i)
	image = (batch[0][i-1]*255).astype('uint8')
	plt.imshow(image)
plt.show()

# izgradnja modela - functional API
inputs = keras.Input(shape=(48,48,3))
x = layers.Conv2D(32, kernel_size=(3,3), padding='same', activation="relu")(inputs)
x = layers.Conv2D(32, kernel_size=(3,3), padding='valid', activation="relu")(x)
x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
x = layers.Dropout(rate = 0.2)(x)
x = layers.Conv2D(64, kernel_size=(3,3), padding='same', activation="relu")(x)
x = layers.Conv2D(64, kernel_size=(3,3), padding='valid', activation="relu")(x)
x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
x = layers.Dropout(rate = 0.2)(x)
x = layers.Conv2D(128, kernel_size=(3,3), padding='same', activation="relu")(x)
x = layers.Conv2D(128, kernel_size=(3,3), padding='valid', activation="relu")(x)
x = layers.MaxPool2D(pool_size=(2,2), strides=(2,2))(x)
x = layers.Dropout(rate = 0.2)(x)
x = layers.Flatten()(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(rate = 0.5)(x)
outputs = layers.Dense(43, activation="softmax")(x)

model = keras.Model(inputs=inputs, outputs=outputs, name="gtsrb_model")

model.summary()

# podesi proces treniranja
model.compile(loss="categorical_crossentropy",
                        optimizer="adam",
                        metrics=["accuracy",])


my_callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
    keras.callbacks.ModelCheckpoint(filepath='checkpoints_gtsrb_cnn_aug/model.{epoch:02d}-{val_loss:.2f}.h5',
                                    save_best_only=True,
                                    monitor = "val_accuracy",
                                    mode="max"),
    keras.callbacks.TensorBoard(log_dir='logs',
                                update_freq=100),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss",
                                    factor = 0.1,
                                    patience = 3,
                                    cooldown = 1)
]

history = model.fit(
    train_generator,
    steps_per_epoch = train_generator.samples // batch_size,
    validation_data = validation_generator,
    validation_steps = validation_generator.samples // batch_size,
    callbacks = my_callbacks,
    epochs=100)

score = model.evaluate(test_generator, batch_size=1, steps=test_generator.samples )
print("Test loss:", score[0])
print("Test accuracy:", score[1])