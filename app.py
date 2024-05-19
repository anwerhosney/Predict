from flask import Flask, render_template, request
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import base64

model = load_model('./model.h5')  # Load pre-trained model
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

app = Flask(__name__, template_folder='./templates', 
static_folder='./static/')
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 1

@app.route('/')
def man():
    return render_template('index.html')

@app.route('/home', methods=['POST'])
def home():
    img = request.files['image']
    img_binary = img.read()

    img_arr = cv2.imdecode(np.frombuffer(img_binary, np.uint8), -1)
    img_arr = cv2.resize(img_arr, (128, 128))
    img_arr = img_arr / 255.0
    img_arr = img_arr.reshape(1, 128, 128, 3)
    prediction = model.predict(img_arr)
    probability_class_1 = prediction[0, 0]

    # Pass prediction and image as a dictionary to the template
    return render_template('prediction.html', data={'image': base64.b64encode(img_binary).decode('utf-8'), 'prediction': probability_class_1})

if __name__ == '__main__':
    app.run(debug=True)