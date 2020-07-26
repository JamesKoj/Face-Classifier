import tensorflow as tf
import PIL
import os
import zipfile
from os import path, getcwd, chdir
import time

def train_happy_sad_model():
    # train_faces_dir = os.path.join()

    DESIRED_ACCURACY = 0.91

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if (logs.get('val_acc') > DESIRED_ACCURACY):
                # Cancels training when desired Validation accuracy is reached to prevent overfitting
                print("\nReached 91% accuracy so cancelling training!")
                saved_model_path = "C:\\DirectoryToStoreModel/" + "my_model3.h5"
                model.save(saved_model_path)
                self.model.stop_training = False

    callbacks = myCallback()

    # This Code Block Defines and Compiles the Model.
    model = tf.keras.models.Sequential([
        # Note the input shape is the desired size of the image 300x300 with 3 bytes color
        # This is the first convolution
        tf.keras.layers.Conv2D(12, (1, 1), activation='relu', input_shape=(300, 300, 1)),
        tf.keras.layers.MaxPooling2D(4, 4),
        tf.keras.layers.Conv2D(16, (1, 1), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(16, (3, 3), activation='relu'),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dense(64, activation=tf.nn.relu),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    from tensorflow.keras.optimizers import RMSprop

    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['acc'])

    # This code block should create an instance of an ImageDataGenerator called train_datagen
    # And a train_generator by calling train_datagen.flow_from_directory

    from tensorflow.keras.preprocessing.image import ImageDataGenerator

    # All images will be rescaled by 1./255
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        "C:\\TrainingSet",  # This is the source directory for training images
        target_size=(300, 300),
        batch_size=32,
        color_mode='grayscale',
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    validation_generator = validation_datagen.flow_from_directory(
        'C:\\ValidationSet',  # This is the source directory for training images
        target_size=(300, 300),  # All images will be resized to 150x150
        batch_size=16,
        color_mode='grayscale',
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')
    # Expected output: 'Found 80 images belonging to 2 classes'

    # This code block should call model.fit_generator and train for
    # a number of epochs.
    # model fitting
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=8,
        epochs=45,
        verbose=1,
        validation_data=validation_generator,
        validation_steps=8,
        callbacks=[callbacks])

    # model fitting



    return history.history['acc'][-1]

train_happy_sad_model()