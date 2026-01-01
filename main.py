import cv2
import numpy as np
import streamlit as st
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image

def load_model():
    model=MobileNetV2(weights='imagenet')
    return model

def preprocess_image(image):
    
    image=np.array(image)
    image=cv2.resize(image,(244,244))
    image=preprocess_input(image)
    image=np.expand_dims(image,axis=0)
    return image
