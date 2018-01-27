import os
from flask import Flask, request, redirect, url_for, render_template

# Local modules
from ml import Prediction, predict
from dog_ml import detect_dog, predict_breed
from human_ml import detect_face
from flask_utils import path_for_file, save_file, UPLOAD_FOLDER

app = Flask(__name__)
# Set upload folder
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Set max allowed size for uploading file
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and save_file(file):
            prediction = predict(path_for_file(file))
            if prediction.subject:
                return redirect(url_for('uploaded_file',
                                        subject=prediction.subject,
                                        filename=file.filename,
                                        breed=prediction.breed))
            else:
                return redirect(url_for('error'))
        else:
            return redirect(request.url)
    return render_template('index.html')

@app.route('/error', methods=['GET', 'POST'])
def error():
    return render_template('error.html')

@app.route('/<subject>/<filename>/<breed>')
def uploaded_file(subject, filename, breed):
    return render_template('results.html',
                           subject=subject,
                           filename=filename,
                           breed=breed)

# Warning
# Without manually calling tf related stuff on main thread
# tf crashes on requests handled by werkzeug, not sure how it's related
print(predict('../my_images/2.jpg'))
print(detect_dog('../my_images/2.jpg'))
print(detect_face('../my_images/2.jpg'))
# End

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the given port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)
