from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
import matplotlib.pyplot as plt

classifier = Sequential()

'''Part 1'''

# First convolution layer and pooling
classifier.add(Convolution2D(16, (3, 3), input_shape=(64, 64, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Dropout(0.2))
classifier.add(BatchNormalization())

classifier.add(Convolution2D(32, (3, 3), activation='relu'))
# Add a dropout layer to prevent the model from overfitting
classifier.add(Dropout(0.2))
# Add a batch normalization so that we get equal probabilities while dealing with the dataset
classifier.add(BatchNormalization())

classifier.add(Convolution2D(64, (3, 3), activation='relu'))
# input_shape is going to be the pooled feature maps from the previous convolution layer
classifier.add(MaxPooling2D(pool_size=(2, 2)))
# Add a batch normalization so that we get equal probabilities while dealing with the dataset
classifier.add(BatchNormalization())
classifier.add(Flatten())

# Add a fully connected layer
classifier.add(Dense(units=3, activation='softmax'))

# categorical_crossentropy for more than 2
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator



train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=20,
    zoom_range=0.05,
    width_shift_range=0.05,
    height_shift_range=0.05,
    shear_range=0.05,
    horizontal_flip=True,
    fill_mode="nearest")

test_datagen = ImageDataGenerator(rescale=1. / 255)

training_set = train_datagen.flow_from_directory('data/train',
                                                 target_size=(64, 64),
                                                 batch_size=30,
                                                 color_mode='rgb',
                                                 class_mode='categorical')

test_set = test_datagen.flow_from_directory('data/test',
                                            target_size=(64, 64),
                                            batch_size=10,
                                            color_mode='rgb',
                                            class_mode='categorical')
classifier.fit_generator(
    training_set,
    steps_per_epoch=336//30,  # No of images in training set
    epochs=10,
    validation_data=test_set,
    validation_steps=90//10)  # No of images in test set

# Saving the model
model_json = classifier.to_json()
with open("model-bw.json", "w") as json_file:
    json_file.write(model_json)
classifier.save_weights('model-bw.h5')
