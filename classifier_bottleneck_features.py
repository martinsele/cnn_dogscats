'''This script goes along the blog post
"Building powerful image classification models using very little data"
from blog.keras.io.
It uses data that can be downloaded at:
https://www.kaggle.com/c/dogs-vs-cats/data
In our setup, we:
- created a data/ folder
- created train/ and validation/ subfolders inside data/
- created cats/ and dogs/ subfolders inside train/ and validation/
- put the cat pictures index 0-999 in data/train/cats
- put the cat pictures index 1000-1400 in data/validation/cats
- put the dogs pictures index 12500-13499 in data/train/dogs
- put the dog pictures index 13500-13900 in data/validation/dogs
So that we have 1000 training examples for each class, and 400 validation examples for each class.
In summary, this is our directory structure:
```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```
'''
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications, Input
from keras import backend as K
from scipy.misc import imresize

# dimensions of our images.
img_width, img_height = 150, 150

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = create_core_model()
    print('Model loaded.')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print('Predict train samples to generate bottleneck features.')
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print('Predict validation samples to generate bottleneck features.')
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)

    print('Save features.')
    np.save(open('bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('bottleneck_features_train.npy', 'rb'))  # shape==(2000, 4, 4, 512)
    print('Train features loaded.')

    train_labels = np.array([0] * (nb_train_samples // 2) + [1] * (nb_train_samples // 2))

    validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
    print('Validation features loaded.')
    validation_labels = np.array([0] * (nb_validation_samples // 2) + [1] * (nb_validation_samples // 2))

    model = create_top_model(train_data.shape[1:])
    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    print('Fit top model')
    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    print('Save model')
    model.save_weights(top_model_weights_path)


def create_top_model(data_shape=[4, 4, 512]):
    model = Sequential()
    model.add(Flatten(input_shape=data_shape))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    return model


def create_core_model():
    input_tensor = Input(shape=(150, 150, 3))
    model = applications.VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

    return model


def classify_one():
    core_model = create_core_model()
    top_model = create_top_model(core_model.output_shape[1:])
    top_model.load_weights(top_model_weights_path)

    # stack on each other
    # model = Model(inputs=core_model.input, outputs=top_model(core_model.output))

    img = load_img('data/train/cats/cat.10831.jpg')  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = imresize(x, input_shape)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    rescaled = x * (1. / 255)

    pred_core = core_model.predict(rescaled)
    prediction = top_model.predict(pred_core)

    # prediction = top_model.predict(rescaled)

    print(prediction)


# save_bottlebeck_features()
# train_top_model()
classify_one()
