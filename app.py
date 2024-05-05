from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
import os
import numpy as np
from PIL import Image
import tensorflow as tf

app = Flask(__name__)

# Configure the path for the uploaded images and the TensorFlow Lite model
app.config['UPLOAD_FOLDER'] = 'static/images'
MODEL_PATH = r'C:\Users\User\Downloads\app\app\models\diabetic_retinopathy_model.tflite'

# Ensure the upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the file's extension is among the allowed ones."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_tflite_model():
    """Load the TensorFlow Lite model and allocate tensors."""
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_tflite_model()

def predict_image(image_path):
    """Perform prediction on the uploaded image using the loaded TensorFlow Lite model."""
    img = Image.open(image_path)
    img = img.resize((224, 224))  # Resize to match the model's expected input
    img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output_data)  # Return the index of the highest probability class

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            prediction = predict_image(filepath)
            # Convert numeric prediction to a label
            labels = {0: "No Diabetic Retinopathy", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative DR"}
            prediction_label = labels.get(prediction, "Unknown class")
            image_url = url_for('static', filename='images/' + filename)
            return render_template('predict.html', image_url=image_url, prediction=prediction, prediction_label=prediction_label)
        else:
            return render_template('predict.html', message="Invalid file type. Please upload an image file.")
    return render_template('predict.html')



if __name__ == '__main__':
    app.run(debug=True, port=5000)
