from flask import Flask, request, render_template, jsonify
import pickle
from PIL import Image
import io
import numpy as np
import cv2
from skimage import io
from skimage.color import rgb2gray
from skimage.feature import hog 
import pickle
from helper import bin_spatial , compute_lbp_features ,color_hist 

app = Flask(__name__)


@app.route('/' , methods = ['GET'])
def home():
    return render_template('indx.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    img_path = "./images/" + imagefile.filename
    imagefile.save(img_path)
    conc_features= []
    file_features = []
    inf = cv2.imread(img_path)
    inf = cv2.cvtColor(inf, cv2.COLOR_BGR2RGB)
    inf = cv2.resize(inf, (256, 256)) 
    hog_image = rgb2gray(inf)

    fd, im = hog(hog_image, orientations=9, pixels_per_cell=(8, 8),
                                cells_per_block=(2, 2), visualize=True)

    file_features = []
    spatial_features = bin_spatial(inf)
    file_features.append(spatial_features)
    lbp_features = compute_lbp_features(inf)
    file_features.append(lbp_features)
    hist_features = color_hist(inf)
    file_features.append(hist_features)
    file_features.append(fd)
    conc_features.append(np.concatenate(file_features))

    file_name = "CarBusModel.pkl"

    with open(file_name, 'rb') as file:
        loaded_model = pickle.load(file)

    # Use the loaded model to make predictions
    predictions = []
    predictions = loaded_model.predict(conc_features)
    pred =  int(predictions[0])

    if pred == 1 :
        result= "car"
    elif pred == 0: 
        result= "bus"
    return render_template('indx.html', prediction = result)

if __name__ == '__main__':
    app.run(debug=True)
