import streamlit as st
from numpy import mean
from PIL import Image
from preprocess import preprocess
from keras.models import load_model



@st.cache
def load_mdl():
    model = load_model("./Model/")
    return model

def predict(model, image):
    prediction = model.predict_classes(image)
    prediction = mean(prediction)
    return prediction


model = load_mdl()

st.title("Skin Myaloma Classification")
st.write("Type of myaloma will be classified based on your uploaded image")
image_file = st.file_uploader("Please upload the image here", type="jpg")

if(image_file is not None):

    img = Image.open(image_file)
    st.write("Make sure this is the intended image")
    st.image(img)

    if(st.button("Continue")):
        preprocessed = preprocess(img)
        prediction = predict(model, preprocessed)
        
        st.text(prediction)



