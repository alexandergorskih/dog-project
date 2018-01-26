from collections import namedtuple
from dog_ml import detect_dog, predict_breed
from human_ml import detect_face

Prediction = namedtuple('Prediction', ['subject', 'img_path', 'breed'])

def predict(img_path):
    subj = detect_dog(img_path) and 'dog' or detect_face(img_path) and 'human'
    breed = predict_breed(img_path)
    return Prediction(subj, img_path, breed)

# Warning
# Without manually calling tf related stuff on main thread
# tf crashes on requests handled by werkzeug, not sure how it's related
print(predict('my_images/2.jpg'))
print(detect_dog('my_images/2.jpg'))
print(detect_face('my_images/2.jpg'))
# End
