import numpy as np
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.utils import np_utils
from keras.preprocessing import image
from tqdm import tqdm
from glob import glob

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

def extract_Xception(tensor):
    from keras.applications.xception import Xception, preprocess_input
    return Xception(weights='imagenet', include_top=False).predict(tensor)

dog_names = [item[20:-1] for item in sorted(glob("dogImages/train/*/"))]

Xception_model = Sequential()
Xception_model.add(GlobalAveragePooling2D(input_shape=(7, 7, 2048), name='GAPLayer'))
Xception_model.add(Dense(133, activation='softmax'))

Xception_model.summary()
Xception_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
Xception_model.load_weights('saved_models/weights.best.Xception.hdf5')

def Xception_predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = extract_Xception(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = Xception_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]

from collections import namedtuple

Prediction = namedtuple('Prediction', ['subject', 'img_path', 'breed'])

def predict(img_path):
    subj = "human"
    breed = Xception_predict_breed(img_path)
    return Prediction(subj, img_path, breed)

print(predict('my_images/2.jpg'))
