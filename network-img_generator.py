from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import TensorBoard
import time

train_data_dir = '/home/simon/Dokumente/Programmieren/KI/Projekte/DeepChemistry/trainingsdaten'
img_height = 64
img_width = 64


epochs = 25
batch_sizes = [16]
dense_layers = [1]
layer_sizes = [512]
conv_layers = [1, 2, 3, 4]





for batch_size in batch_sizes:

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        validation_split=0.2)  # set validation split

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse',
        subset='training')  # set as training data

    validation_generator = train_datagen.flow_from_directory(
        train_data_dir,  # same directory as training data
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='sparse',
        subset='validation')  # set as validation data

    for dense_layer in dense_layers:
        for layer_size in layer_sizes:
            for conv_layer in conv_layers:

                NAME = "{}-conv-{}-nodes-{}-dense-{}-batch-{}".format(conv_layer, layer_size, dense_layer, batch_size,int(time.time()))

                print(NAME)

                model = Sequential()

                model.add(Conv2D(layer_size, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)))
                model.add(MaxPooling2D(pool_size=(2, 2)))
                model.add(Dropout(0.25))

                for l in range(conv_layer - 1):
                    model.add(Conv2D(layer_size, (3, 3), activation="relu"))
                    model.add(MaxPooling2D(pool_size=(2, 2)))
                    model.add(Dropout(0.25))

                model.add(Flatten())

                for _ in range(dense_layer):
                    model.add(Dense(layer_size, activation="relu"))

                model.add(Dense(21, activation="softmax"))

                tensorboard = TensorBoard(log_dir="logs/v2/{}".format(NAME))


                model.compile(loss="sparse_categorical_crossentropy",
                              optimizer="rmsprop",
                              metrics=['sparse_categorical_accuracy'])

                model.fit(
                    train_generator,
                    steps_per_epoch = train_generator.samples // batch_size,
                    validation_data = validation_generator,
                    validation_steps = validation_generator.samples // batch_size,
                    epochs = epochs,
                    callbacks=([tensorboard]))



