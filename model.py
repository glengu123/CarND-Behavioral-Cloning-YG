import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Flatten, Dropout, Lambda,Activation
from keras.layers.convolutional import Convolution2D, Cropping2D
from keras.layers.pooling import MaxPooling2D 
from keras.models import Sequential
from keras.layers import Flatten, Dense

samples = []
def read_img(csv_path):
    with open(csv_path) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            samples.append(line)
        return samples[1:]
csv_samples = read_img('./data/driving_log.csv')

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(csv_samples, test_size=0.2)
        
def flipped(image):
    return np.fliplr(image)

import sklearn
def generator(samples, batch_size=128):
    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]
            images = []
            angles = []
            data_path = './data/IMG'
            for batch_sample in batch_samples:
                center_img = cv2.imread(data_path + batch_sample[0].split('IMG')[1])
                center_img_flip = flipped(center_img)
                left_img = cv2.imread(data_path + batch_sample[1].split('IMG')[1])
                left_img_flip = flipped(left_img)
                right_img = cv2.imread(data_path + batch_sample[2].split('IMG')[1])
                right_img_flip = flipped(right_img)
                center_angle = float(batch_sample[3])
                center_angle_flip = - center_angle
                left_angle = float(batch_sample[3]) + 0.2
                left_angle_flip = - left_angle
                right_angle = float(batch_sample[3]) - 0.2
                right_angle_flip = - right_angle
                images.append(center_img)
                images.append(left_img)
                images.append(right_img)
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
                images.append(center_img_flip)
                images.append(left_img_flip)
                images.append(right_img_flip)
                angles.append(center_angle_flip)
                angles.append(left_angle_flip)
                angles.append(right_angle_flip)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=128)
validation_generator = generator(validation_samples, batch_size=128)

model = Sequential()
model.add(Cropping2D(cropping = ((70, 25), (0, 0)), input_shape = (160, 320, 3)))
model.add(Lambda(lambda x: (x / 127.5) - 1., input_shape = (65, 320, 3)))
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(100))
model.add(Activation('elu'))
model.add(Dropout(0.2))
model.add(Dense(50))
model.add(Activation('elu'))
model.add(Dropout(0.2))
model.add(Dense(10))
model.add(Activation('elu'))
model.add(Dropout(0.2))
model.add(Dense(1))


model.compile(loss='mse', optimizer='adam')
from keras.models import Model
import matplotlib.pyplot as plt

history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples) *6, validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples) * 6, 
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
#print(history_object.history.keys())
#plt.rcParams['figure.figsize'] = (16, 9)
### plot the training and validation loss for each epoch
#plt.plot(history_object.history['loss'])
#plt.plot(history_object.history['val_loss'])
#plt.title('model mean squared error loss')
#plt.ylabel('mean squared error loss')
#plt.xlabel('epoch')
#plt.legend(['training set', 'validation set'], loc='upper right')
#plt.show()
model.save('model-2.h5')