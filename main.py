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
    image=cv2.resize(image,(224,224))
    image=preprocess_input(image)
    image=np.expand_dims(image,axis=0)
    return image


def clasify_image(model,image):
    try:
        processed_img=preprocess_image(image)
        preds=model.predict(processed_img)
        decoded_preds=decode_predictions(preds,top=5)[0]
        return decoded_preds
    except Exception as e:
        st.error(f"Error in classification: {str(e)}")
        return None

def main():   
    st.set_page_config(page_title="Image Classifier",layout="centered",page_icon="🤖")
    st.title("🤖 Image Classifier")
    st.markdown("Upload an image to classify it using MobileNetV2 model.")
    @st.cache_resource
    def load_cached_model():
        return load_model()
    model=load_cached_model()
    uploaded_file=st.file_uploader("Choose an image...",type=["jpg","jpeg","png"])
    if uploaded_file is not None:
        image=st.image(uploaded_file,caption="Uploaded Image",use_column_width=True)
        btn=st.button("Classify Image")

        if btn:
            with st.spinner("Classifying..."):
                image=Image.open(uploaded_file)
                predictions=clasify_image(model,image)
                if predictions:
                    st.subheader("Top Predictions:")
                    for _,label,score in predictions:
                        st.write(f"{label}: {score:.2%}")

if __name__=="__main__":
    main()




