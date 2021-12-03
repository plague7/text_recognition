import os

import streamlit as st

from tensorflow import keras
from tensorflow.keras.utils import *
import tensorflow as tf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image
from streamlit_drawable_canvas import st_canvas

from random import randrange

#MODEL_DIR = os.path.join(os.path.dirname('__file__'), 'first_model.h5')

### Visualization function ###
def viz_num(num):
    #Reshape the 768 values to a 28x28 image
    image = X_raw_final[num].reshape([28,28])
    fig = plt.figure(figsize=(1, 1))
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.axis("off")
    fig.show()
    return fig

### Load / Preprocess / Predict test.csv dataset ###

model = keras.models.load_model(r"C:\Users\Simplon\Desktop\Travaux python\CNN exercice data\first_model.h5")
# MODEL_DIR

#MODEL_DIR = os.path.join(os.path.dirname('__file__'), 'first_model.h5')
#model = keras.models.load_model(MODEL_DIR)

###### STREAMLIT ######

### Changing style ###
st.markdown(""" <style> img {
width:150px !important; height:150px;} 
</style> """, unsafe_allow_html=True)

### Header ###
st.title('Digit Prediction App')
st.header('Which prediction tool to use?')

### Selectbox ###

### Randomizing Tool View ###
if st.selectbox('Tools', ['Randomizing Tool', 'Drawing Tool']) == 'Randomizing Tool':
    ### Upload test dataset ###
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        data_test = pd.read_csv(uploaded_file)
        X_raw_final = data_test.values
        X_test_final = data_test.values.reshape(data_test.shape[0], 28, 28, 1)
        pred_testing = model.predict(X_test_final)
        pred_testing = np.argmax(pred_testing, axis=1)

        ### Display button for prediction ###
        if st.button('Predict a random image from our dataframe'):
            random_number = randrange(28000)
            st.write('Picture number ' + str(random_number))
            st.write('Predicted number : ' + str(pred_testing[random_number]))
            viz = viz_num(random_number)
            st.pyplot(viz) 

### Drawing Tool View ###               
else:
    st.write('Draw a number and let the app makes a prediction')

    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 8)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#fff")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000")
    bg_image = st.sidebar.file_uploader("Background image:", type=["png", "jpg"])
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "line", "rect", "circle", "transform")
    )
    realtime_update = st.sidebar.checkbox("Update in realtime", True)

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.open(bg_image) if bg_image else None,
        update_streamlit=realtime_update,
        height=120,
        width=120,
        drawing_mode=drawing_mode,
        key="canvas",
    )

    # Resize drawn image and predict it
    if canvas_result.image_data is not None:
        number_drawn = canvas_result.image_data
        if st.button('Predict your drawn number'):
            image = Image.fromarray((number_drawn[:, :, 0]).astype(np.uint8))
            image = image.resize((28, 28))
            image = image.convert('L')
            image = (tf.keras.utils.img_to_array(image)/255)
            image = image.reshape(1,28,28,1)
            x_2 = tf.convert_to_tensor(image)
            pred = model.predict(x_2)
            pred = np.argmax(pred, axis=1)
            st.write('Predicted number : ' + str(pred))