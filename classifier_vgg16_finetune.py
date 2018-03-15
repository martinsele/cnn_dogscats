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

from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense
from keras import applications, Input
from keras import backend as K
from scipy.misc import imresize
import os

# path to the model weights files.
top_model_weights_path = 'bottleneck_fc_model.h5'
fine_tuned_model_weights_path = 'fine_tuned_weights.h5'
# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 50
batch_size = 32

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)


def create_limited_train_model():
    # build the VGG16 network
    base_model = applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    top_model = Sequential()
    top_model.add(Flatten(input_shape=base_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # note that it is necessary to start with a fully-trained classifier, including the top classifier,
    # in order to successfully do fine-tuning
    top_model.load_weights(top_model_weights_path)

    # add the model on top of the convolutional base
    # base_model.add(top_model)
    model = Model(inputs=base_model.input, outputs=top_model(base_model.output))
    # model = Sequential()
    # for layer in base_model.layers:
    #     model.add(layer)
    # model.add(top_model)

    # set the first 25 layers (up to the last conv block) to non-trainable (weights will not be updated)
    for layer in model.layers[:15]:
        layer.trainable = False

    return model

def fine_tune():
    model = create_limited_train_model()

    # compile the model with a SGD/momentum optimizer and a very slow learning rate.
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    # prepare data augmentation configuration
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='binary')

    print('Fine tuning...')
    # fine-tune the model
    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples//batch_size,
        # samples_per_epoch=nb_train_samples,
        epochs=epochs,
        validation_data=validation_generator,
        nb_val_samples=nb_validation_samples,
        workers=6)

    model.save_weights(fine_tuned_model_weights_path)
    print('DONE...')


def classify_dir():
    model = create_limited_train_model()
    model.load_weights(fine_tuned_model_weights_path)

    # stack on each other
    # model = Model(inputs=core_model.input, outputs=top_model(core_model.output))

    directory = os.fsencode('data/validation/cats')
    i = 0
    classif = 0
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename.endswith(".asm") or filename.endswith(".py"):
            continue
        else:
            print(i)
            i += 1
            file_n = os.path.join(directory, file)
            correct = classify_one(model=model, file_name=file_n, correct_class=0)
            if correct == 0:
                print('Miss classified %s', file_n)
                classif += 1
                # break
            print('Incorrect %d / %d', classif, i)
    print('DONE')


def classify_one(model, file_name, correct_class):
    img = load_img(file_name)  # this is a PIL image
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = imresize(x, input_shape)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    rescaled = x * (1. / 255)

    prediction = model.predict(rescaled)
    print(prediction)
    if round(prediction.tolist()[0][0]) != correct_class:
        return 0
    return 1


if __name__=='__main__':
    # fine_tune()
    classify_dir()
