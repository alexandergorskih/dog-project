# ML PART - will refactor after bugfix

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

# WEBSERVER PART
import os
from flask import Flask, request, redirect, url_for, render_template, Response
from flask import send_from_directory
from werkzeug.utils import secure_filename
import time

UPLOAD_FOLDER = './static/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(path)
            prediction = predict(path)
            return redirect(url_for('uploaded_file',
                                    subject=prediction.subject,
                                    filename=filename,
                                    breed=prediction.breed))
    return render_template('index.html')

@app.route('/<subject>/<filename>/<breed>')
def uploaded_file(subject, filename, breed):
    return render_template('results.html', subject=subject, filename=filename, breed=breed)

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the givn port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)
