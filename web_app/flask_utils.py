import os
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = './static/'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def path_for_file(file):
    filename = secure_filename(file.filename)
    return os.path.join(UPLOAD_FOLDER, filename)

def save_file(file):
    if file and allowed_file(file.filename):
        file.save(path_for_file(file))
        return True
    else:
        return False
