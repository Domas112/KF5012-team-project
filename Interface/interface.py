import streamlit as st
import numpy as np
from PIL import Image
from preprocess import preprocess
from keras.models import load_model

@st.cache
def load_mdl():
    model = load_model("./Model/")
    return model

def predict(model, image):
    image = preprocess(image)
    prediction = model.predict(image)
    return prediction

model = load_mdl()

st.title("Skin Myeloma Classification")
image_file = st.file_uploader("Please upload the image here", type=["jpg", "png", "jpeg"])

if(image_file is not None):

    img = Image.open(image_file)
    st.write("Make sure this is the intended image")
    st.image(img)

    if(st.button("Continue")):
        prediction = predict(model, img)
        st.write("The type of myeloma is predicted to be ")
        
        if(prediction):
            st.header("Malignant")
        
        else:
            st.header("Benign")


