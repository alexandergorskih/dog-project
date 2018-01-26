import os
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
from ml import Prediction, predict

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

app = Flask(__name__)
# Set upload folder
app.config['UPLOAD_FOLDER'] = './static/'
# Set max allowed size for uploading file
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def path_for_file(file):
    filename = secure_filename(file.filename)
    return os.path.join(app.config['UPLOAD_FOLDER'], filename)

def save_file(file):
    if file and allowed_file(file.filename):
        file.save(path_for_file(file))
        return True
    else:
        return False

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files.get('file')
        if file and save_file(file):
            prediction = predict(path_for_file(file))
            if prediction:
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

if __name__ == "__main__":
	#decide what port to run the app in
	port = int(os.environ.get('PORT', 5000))
	#run the app locally on the given port
	app.run(host='0.0.0.0', port=port)
	#optional if we want to run in debugging mode
	#app.run(debug=True)
