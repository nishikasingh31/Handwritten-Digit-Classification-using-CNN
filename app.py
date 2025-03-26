import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model # type: ignore
import matplotlib.pyplot as plt
import os


st.set_page_config(
    page_title="Digit Classifier",
    page_icon="🔢",
    layout="centered"
)

@st.cache_resource
def load_model_cached():
    try:
        model = load_model('model/saved_models/digit_recognizer.h5', compile=False)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model
    except Exception as e:
        st.error(f"Model loading failed: {str(e)}")
        return None

model = load_model_cached()

st.title("Handwritten Digit Classifier")


# Sidebar (now comes after set_page_config)
with st.sidebar:
    st.header("About")
    st.markdown("""
    - **Model**: CNN trained on MNIST
    - **Accuracy**: ~99% on test data
    - **Input**: Single digit (0-9) images
    """)


def preprocess_image(uploaded_image):
    """Convert uploaded image to MNIST format"""
    try:
        # Convert to grayscale and resize
        img = Image.open(uploaded_image).convert('L').resize((28, 28))
        
        # Convert to array and invert colors
        img_array = np.array(img)
        img_array = 255 - img_array  # Invert colors
        
        # Normalize and reshape
        img_array = (img_array / 255.0).astype(np.float32)
        return img_array.reshape(1, 28, 28, 1)  # Add batch dimension
    except Exception as e:
        st.error(f"Image processing error: {str(e)}")
        return None

uploaded_file = st.file_uploader(
    "Upload digit image",
    type=["png", "jpg", "jpeg"],
    help="Image should contain a single digit (0-9)"
)

if uploaded_file and model:
    # Show original and preprocessed images side-by-side
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown('<h3 style="font-size:18px;">Your Image</h3>', unsafe_allow_html=True)
        img = Image.open(uploaded_file).convert('L')
        st.image(img, width=395)

    with col2:
        st.markdown('<h3 style="font-size:18px;">Preprocessed Image</h3>', unsafe_allow_html=True)
        processed_img = preprocess_image(uploaded_file)
        if processed_img is not None:
            fig, ax = plt.subplots()
            ax.imshow(processed_img[0, :, :, 0], cmap='gray')
            st.pyplot(fig)

    if st.button("Classify Digit"):
        if processed_img is not None:
            try:
                prediction = model.predict(processed_img)
                digit = np.argmax(prediction)
                with col3:
                    st.markdown('<h3 style="font-size:18px;">Results</h3>', unsafe_allow_html=True)
                    st.metric("Predicted Digit", digit)
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
