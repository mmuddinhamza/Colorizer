# coding: utf-8



# import the necessary packages
import numpy as np
import cv2
import streamlit as st
from PIL import Image
import os
import io

st.set_page_config(page_title="Image Colorizer", page_icon="🎨", layout="wide")


def colorizer(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    # load our serialized black and white colorizer model and cluster
    # center points from disk
    #Note: Please take in account the directories of your local system.
    DIR = os.getcwd()
    prototxt = os.path.join(DIR, r"models/models_colorization_deploy_v2.prototxt")
    points = os.path.join(DIR, r"models/pts_in_hull.npy")
    model = os.path.join(DIR, r"models/colorization_release_v2.caffemodel")
    net = cv2.dnn.readNetFromCaffe(prototxt, model)
    pts = np.load(points)
    # add the cluster centers as 1x1 convolutions to the model
    class8 = net.getLayerId("class8_ab")
    conv8 = net.getLayerId("conv8_313_rh")
    pts = pts.transpose().reshape(2, 313, 1, 1)
    net.getLayer(class8).blobs = [pts.astype("float32")]
    net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]
    # scale the pixel intensities to the range [0, 1], and then convert the image from the BGR to Lab color space
    scaled = img.astype("float32") / 255.0
    lab = cv2.cvtColor(scaled, cv2.COLOR_RGB2LAB)
    # resize the Lab image to 224x224 (the dimensions the colorization
    #network accepts), split channels, extract the 'L' channel, and then perform mean centering
    resized = cv2.resize(lab, (224, 224))
    L = cv2.split(resized)[0]
    L -= 50
    # pass the L channel through the network which will *predict* the 'a' and 'b' channel values
    net.setInput(cv2.dnn.blobFromImage(L))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    # resize the predicted 'ab' volume to the same dimensions as our input image
    ab = cv2.resize(ab, (img.shape[1], img.shape[0]))
    # grab the 'L' channel from the *original* input image (not the
    # resized one) and concatenate the original 'L' channel with the predicted 'ab' channels
    L = cv2.split(lab)[0]
    colorized = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    # convert the output image from the Lab color space to RGB, then clip any values that fall outside the range [0, 1]
    colorized = cv2.cvtColor(colorized, cv2.COLOR_LAB2RGB)
    colorized = np.clip(colorized, 0, 1)
    # the current colorized image is represented as a floating point
    # data type in the range [0, 1] -- let's convert to an unsigned 8-bit integer representation in the range [0, 255]
    colorized = (255 * colorized).astype("uint8")
    # Return the colorized images
    return colorized

##########################################################################################################
    
st.title("🎨 Colorize your Black & White Image")
st.markdown("Transform your old black and white photos into vibrant color images using deep learning.")

st.sidebar.header("Upload Settings")
file = st.sidebar.file_uploader("Please upload an image file", type=["jpg", "jpeg", "png"])

if file is None:
    st.info("👈 Please upload an image file from the sidebar to get started.")
else:
    image = Image.open(file)
    img = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, width='stretch')
        
    with col2:
        st.subheader("Colorized Image")
        with st.spinner("Colorizing... Please wait."):
            color = colorizer(img)
        st.image(color, width='stretch')
        
        color_image_pil = Image.fromarray(color)
        buf = io.BytesIO()
        color_image_pil.save(buf, format="JPEG")
        st.download_button(
            label="Download Colorized Image",
            data=buf.getvalue(),
            file_name="colorized.jpg",
            mime="image/jpeg",
            width='stretch'
        )
        
    st.success("Colorization complete!")
