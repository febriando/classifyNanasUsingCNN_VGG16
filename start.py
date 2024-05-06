from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from skimage.feature import graycomatrix, graycoprops
from skimage.io import imread
from skimage.color import rgb2gray
from skimage import img_as_ubyte
from keras_preprocessing.image import load_img
from keras_preprocessing.image import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from keras.applications.mobilenet_v2 import decode_predictions
from keras.applications.mobilenet_v2 import MobileNetV2
import tensorflow as tf
from tensorflow import keras
from skimage import transform, io
import numpy as np
import cv2
import os
from PIL import Image
from datetime import datetime
# from keras.preprocessing import image
import keras.utils as image
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load model for prediction
modelvgg = load_model("model.h5")

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
UPLOAD_FOLDER2 = 'static/uploads2/'
app.config['UPLOAD_FOLDER2'] = UPLOAD_FOLDER2
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'tiff', 'webp', 'jfif'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_texture_features(img):
    # Convert image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create GLCM from the image
    glcm = graycomatrix(img, [1], [0], levels=256, symmetric=True, normed=True)

    # Calculate GLCM properties
    contrast = graycoprops(glcm, 'contrast')[0][0]
    dissimilarity = graycoprops(glcm, 'dissimilarity')[0][0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0][0]
    energy = graycoprops(glcm, 'energy')[0][0]
    correlation = graycoprops(glcm, 'correlation')[0][0]

    return contrast, dissimilarity, homogeneity, energy, correlation

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("cnn.html")

@app.route("/classification", methods=['GET', 'POST'])
def classification():
    return render_template("classifications.html")

@app.route('/training', methods=['POST'])
def train():
    train_dir = request.form['train_dir']
    train_dir = train_dir.replace("\\", "/")

    # Lakukan pelatihan dataset di sini menggunakan train_dir

    return render_template('training.html', position='training', train_dir=train_dir, success='success')

@app.route('/submit', methods=['POST'])
def predict():
    if 'file' not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp
    
    files = request.files.getlist('file')
    filename = "temp_image.png"
    errors = {}
    success = False
    
    for file in files:
        if file and allowed_file(file.filename):
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            success = True
        else:
            errors["message"] = 'File type of {} is not allowed'.format(file.filename)
            
    if 'file2' not in request.files:
        resp = jsonify({'message': 'No image in the request'})
        resp.status_code = 400
        return resp
    
    files2 = request.files.getlist('file2')
    filename2 = "temp_image.png"
    errors2 = {}
    success2 = False
            
    for file2 in files2:
        if file2 and allowed_file(file2.filename):
            file2.save(os.path.join(app.config['UPLOAD_FOLDER2'], filename2))
            success2 = True
        else:
            errors2["message"] = 'File type of {} is not allowed'.format(file.filename)

    if not success:
        resp = jsonify(errors)
        resp.status_code = 400
        return resp
    
    # Get first image
    img_url = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    # Image Processing 1
    img = cv2.imread(img_url)
    contrast, dissimilarity, homogeneity, energy, correlation = extract_texture_features(img)
    class_names = ['Belum Matang', 'Matang', 'Setengah Matang']
    img = image.load_img(img_url, target_size=(150, 150, 3))
    x = image.img_to_array(img)
    x = x / 127.5 - 1
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])

    # Classify image 1
    texture_features = np.array([contrast, dissimilarity, homogeneity, energy, correlation])
    texture_features = np.expand_dims(texture_features, axis=0)
    images_with_features = [images, texture_features]
    prediction_array_vgg = modelvgg.predict(images_with_features[0])
    predicted_class_vgg = class_names[np.argmax(prediction_array_vgg[0])],
    confidence_vgg = '{:2.0f}%'.format(100 * np.max(prediction_array_vgg[0]))
    
    # Get second image
    img_url2 = os.path.join(app.config['UPLOAD_FOLDER2'], filename2)
    
    # Image Processing 2
    img2 = cv2.imread(img_url2)
    contrast2, dissimilarity2, homogeneity2, energy2, correlation2 = extract_texture_features(img2)
    class_names2 = ['Belum Matang', 'Matang', 'Setengah Matang']
    img2 = image.load_img(img_url2, target_size=(150, 150, 3))
    x2 = image.img_to_array(img2)
    x2 = x2 / 127.5 - 1
    x2 = np.expand_dims(x2, axis=0)
    images2 = np.vstack([x2])
    
    # Classify image 2
    texture_features2 = np.array([contrast2, dissimilarity2, homogeneity2, energy2, correlation2])
    texture_features2 = np.expand_dims(texture_features2, axis=0)
    images_with_features2 = [images2, texture_features2]
    prediction_array_vgg2 = modelvgg.predict(images_with_features2[0])
    predicted_class_vgg2 = class_names2[np.argmax(prediction_array_vgg2[0])],
    confidence_vgg2 = '{:2.0f}%'.format(100 * np.max(prediction_array_vgg2[0]))
    
    # fix kematangan
    if (predicted_class_vgg[0]=='Matang') & (predicted_class_vgg2[0]=='Matang'):
        hasil = 'Matang'
    elif (predicted_class_vgg[0]=='Matang') & (predicted_class_vgg2[0]=='Belum Matang'):
        hasil = 'Setengah Matang'
    elif (predicted_class_vgg[0]=='Matang') & (predicted_class_vgg2[0]=='Setengah Matang'):
        hasil = 'Matang'
    elif (predicted_class_vgg[0]=='Belum Matang') & (predicted_class_vgg2[0]=='Belum Matang'):
        hasil = 'Belum Matang'
    elif (predicted_class_vgg[0]=='Belum Matang') & (predicted_class_vgg2[0]=='Setengah Matang'):
        hasil = 'Belum Matang'
    elif (predicted_class_vgg[0]=='Belum Matang') & (predicted_class_vgg2[0]=='Matang'):
        hasil = 'Setengah Matang'
    elif (predicted_class_vgg[0]=='Setengah Matang') & (predicted_class_vgg2[0]=='Setengah Matang'):
        hasil = 'Setengah Matang'
    elif (predicted_class_vgg[0]=='Setengah Matang') & (predicted_class_vgg2[0]=='Matang'):
        hasil = 'Matang'
    else:
        hasil = 'Belum Matang'
    
    # fix confidence kedua gambar
    confidence1 = confidence_vgg.replace("%", "")
    confidence2 = confidence_vgg2.replace("%", "")
    confidence_fix = (int(confidence1) + int(confidence2))/2
    persen = str(confidence_fix) + "%"
    
    # Render the result template with the predicted class, confidence, and image URL
    return render_template("classifications.html", predicted_class_vgg=predicted_class_vgg, confidence_vgg=confidence_vgg, img_url=img_url, 
                           predicted_class_vgg2=predicted_class_vgg2, confidence_vgg2=confidence_vgg2, img_url2=img_url2,
                           hasil=hasil, confidence_fix=persen)
    #response_data = {
    #    'predicted_class_vgg': predicted_class_vgg,
    #    'confidence_vgg': confidence_vgg,
    #}

    # Return the response
    #resp = jsonify(response_data)
    #resp.status_code = 200
    #return resp

if __name__ == '__main__':
    app.run(debug=True)
