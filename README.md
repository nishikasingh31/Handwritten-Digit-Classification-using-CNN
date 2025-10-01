# Handwritten Digit Classification Using CNN

This project demonstrates a **Convolutional Neural Network (CNN)** trained on the **MNIST dataset** to classify handwritten digits.  
It features an interactive **Streamlit web application**, allowing users to draw digits and get real-time predictions.

## 🚀 Features

- High accuracy digit classification on MNIST dataset.  
- Interactive Streamlit GUI for drawing and predicting digits.  
- Model training, saving, and loading for inference.  
- Easy deployment via Streamlit Cloud.

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Deep Learning Framework:** TensorFlow, Keras  
- **Web Framework:** Streamlit  
- **Dataset:** MNIST  

## 📈 Model Performance

The CNN model was trained on the MNIST dataset and achieves:

- **Training Accuracy:** 99.3%  
- **Test Accuracy:** 99.1%  

This demonstrates the model’s high reliability in classifying handwritten digits.

## 🧪 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/nishikasingh31/Handwritten-Digit-Classification-using-CNN.git
cd Handwritten-Digit-Classification-using-CNN
```
2. Install Dependencies
```bash
Copy code
pip install -r requirements.txt
```
3. Train the Model
```bash
Copy code
python src/main.py
```
-Loads and preprocesses MNIST dataset. <br>
-Defines and trains the CNN model. <br>
-Evaluates performance and saves the trained model to model/model.h5. <br>

4. Run the Streamlit App
```bash
Copy code
streamlit run src/app.py
```
-Opens a browser window with the app. <br>
-Draw digits on the canvas and get predictions instantly. 

## 📊 Example
[cnn.pdf](https://github.com/user-attachments/files/19464876/cnn.pdf)

## 🌐 Live Demo
Try the live Streamlit app here: <br>
[Handwritten Digit Classification App](https://nishikasingh31-handwritten-digit-classification-using-cnn.streamlit.app/) <br>

[Project Demo](https://github.com/user-attachments/assets/4582168a-db5e-4b27-a38b-8bcbd789a75d)


