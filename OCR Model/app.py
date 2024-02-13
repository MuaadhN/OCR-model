from flask import Flask, render_template, request
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from keras.models import load_model
import joblib
import matplotlib
matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt
from flask import Flask, render_template, request
import base64
from io import BytesIO


app = Flask(__name__)

# Load the trained model and LabelBinarizer
model = load_model('characters_detection.h5')  # Update with your model file path
LB = joblib.load('label_binarizer.pkl')  # Update with your LabelBinarizer file path

def sort_contours(cnts, method="left-to-right"):
    reverse = False
    i = 0
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
    key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)

def get_letters(file_path):
    letters = []

    # Load the image in color
    image = cv2.imread(file_path)

    if image is None:
        print(f"Unable to read the image at path: {file_path}")
        return letters

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Thresholding and dilation
    ret, thresh1 = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(thresh1, None, iterations=2)

    # Find contours
    cnts = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sort_contours(cnts, method="left-to-right")[0]

    # Loop over the contours
    for c in cnts:
        if cv2.contourArea(c) > 100 and cv2.contourArea(c) < 100000:  # Adjust the threshold as needed
            (x, y, w, h) = cv2.boundingRect(c)
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Extract the region of interest (ROI) and process it
            roi = gray[y:y + h, x:x + w]
            thresh = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

            # Resize the thresholded image to match the input size expected by the model
            thresh = cv2.resize(thresh, (32, 32), interpolation=cv2.INTER_CUBIC)

            # Normalize and preprocess the image for prediction
            thresh = thresh.astype("float32") / 255.0
            thresh = np.expand_dims(thresh, axis=-1)
            thresh = thresh.reshape(1, 32, 32, 1)

        # Make predictions using the provided model
            ypred = model.predict(thresh)

        # Inverse transform the predictions using the provided LabelBinarizer (LB)
            ypred = LB.inverse_transform(ypred)
            [predicted_label] = ypred

            # Append the predicted letter to the list
            letters.append(predicted_label)

# Convert the processed image to base64 for displaying in HTML
    _, buffer = cv2.imencode('.png', cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    image_base64 = base64.b64encode(buffer).decode('utf-8')

# Return the list of predicted letters and the base64-encoded image
    return letters, image_base64



def segment(img, n_colors=2):
    X = img.reshape((-1, 1))
    
    km = KMeans(n_clusters=n_colors)
    km.fit(X)
    
    cen = km.cluster_centers_
    lbl = km.labels_
    
    return cen[lbl].reshape(img.shape)

def get_word(letter):
    word = "".join(letter)
    return word

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return render_template('index.html', message='No file part')

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', message='No selected file')

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

    # Perform image segmentation on the uploaded image
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            return render_template('index.html', message='Unable to read the image')

        segmented = segment(img)

        # Process the segmented image and get the recognized word
        letters, image_base64 = get_letters(file_path)
        word = get_word(letters)

        return render_template('index.html', message='Processed image', word=word, image_base64=image_base64)

if __name__ == '__main__':
    app.run(debug=True, threaded=True)

