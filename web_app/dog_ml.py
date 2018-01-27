import numpy as np
from glob import glob
from keras.applications.resnet50 import preprocess_input
from ml_utils import load_model, path_to_tensor

# Took this strange approach with hardcoding path length from notebook
# Hope there are more elegant ways to do it in python
# But for now I better save some time)
dog_names = [item[23:-1] for item in sorted(glob('../dogImages/train/*/'))]

dog_detector = load_model('dog_detector')
bottleneck = load_model('bottleneck')
loaded_model = load_model('model')

def predict_dog_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(dog_detector.predict(img))

def detect_dog(img_path):
    prediction = predict_dog_labels(img_path)
    return ((prediction <= 268) & (prediction >= 151))

def predict_breed(img_path):
    # extract bottleneck features
    bottleneck_feature = bottleneck.predict(path_to_tensor(img_path))
    # obtain predicted vector
    predicted_vector = loaded_model.predict(bottleneck_feature)
    # return dog breed that is predicted by the model
    return dog_names[np.argmax(predicted_vector)]
