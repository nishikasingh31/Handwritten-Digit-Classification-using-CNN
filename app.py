import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
from streamlit_drawable_canvas import st_canvas # type: ignore

# Load the trained CNN model
model = load_model("C:/Users/nishi/project/8th Major/main/digit_classifier.h5")

# Function to predict digit
def predict_digit(image):
    image = image.resize((28, 28)).convert('L')  # Convert to grayscale
    image = np.array(image, dtype=np.float32) / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)

    prediction = model.predict(image)
    return np.argmax(prediction)

# Streamlit page title
st.title("Handwritten Digit Recognition")

# Create a drawing canvas
canvas_result = st_canvas(
    width=300,
    height=300,
    stroke_width=5,
    stroke_color="black",
    background_color="white",
    drawing_mode="freedraw",
    key="canvas",
)

# Display instructions
st.write("Draw a digit on the canvas above.")

# If there is image data on the canvas, predict the digit
if canvas_result.image_data is not None:
    # Convert canvas image to PIL image
    image = Image.fromarray(canvas_result.image_data.astype("uint8"))

    # Button to predict digit
    if st.button("Predict"):
        digit = predict_digit(image)
        st.write(f'Predicted Digit: {digit}')

# Button to clear the canvas
if st.button("Clear Canvas"):
    st.experimental_rerun()  # Reset the canvas
