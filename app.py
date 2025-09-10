from flask import Flask, render_template, request
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

# Load the trained model
model = load_model("C:\\Users\\srikr\\OneDrive\\Desktop\\SkinCancer\\skincancer_model.h5")

# Mapping of predicted class to skin cancer type
class_mapping = {0: 'type_0', 1: 'type_1', 2: 'type_2', 3: 'type_3', 4: 'type_4', 5: 'type_5', 6: 'type_6'}

# Assuming you have a mapping from type to name
type_to_name_mapping = {
    'type_0': 'Basal cell carcinoma (BCC)',
    'type_1': 'Benign keratosis-like lesions (BKL)',
    'type_2': 'Dermatofibroma (DF)',
    'type_3': 'Melanoma (MEL)',
    'type_4': 'Melanocytic nevi (NV)',
    'type_5': 'Vascular lesions (VASC)',
    'type_6': 'Other'
}

def predict_skin_cancer(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (250, 250))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    predictions = model.predict(img)
    predicted_class = np.argmax(predictions)
    predicted_skin_cancer_name = type_to_name_mapping[class_mapping[predicted_class]]

    return predicted_skin_cancer_name

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')

        file = request.files['file']

        if file.filename == '':
            return render_template('index.html', error='No selected file')

        if file:
            image_path = "static/input_image.jpg"
            file.save(image_path)
            predicted_skin_cancer = predict_skin_cancer(image_path)  # Predict the skin cancer type
            return render_template('index.html', result=predicted_skin_cancer, image_path=image_path)

    return render_template('index.html', error=None, result=None, image_path=None)

if __name__ == '__main__':
    app.run(debug=True)
