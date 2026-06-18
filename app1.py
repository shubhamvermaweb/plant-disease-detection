# 📦 Import libraries
import streamlit as st
import tensorflow as tf
import numpy as np
from keract import get_activations
import matplotlib.pyplot as plt

# 🔍 Model prediction function
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # batch format
    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)
    return predicted_index, confidence, model, input_arr

# 🧠 Layer-wise activation viewer
def show_layer_activations(model, image_array):
    activations = get_activations(model, image_array)
    for layer_name, activation in activations.items():
        st.markdown(f"### 🔍 Layer: `{layer_name}`")
        if activation.ndim == 4:  # Conv2D
            num_filters = min(activation.shape[-1], 6)
            fig, axes = plt.subplots(1, num_filters, figsize=(15, 5))
            for i in range(num_filters):
                axes[i].imshow(activation[0, :, :, i], cmap='viridis')
                axes[i].axis('off')
                axes[i].set_title(f'Filter {i}')
            st.pyplot(fig)
        elif activation.ndim == 2:  # Dense
            st.bar_chart(activation[0])

def apply_global_styling():
    st.markdown("""
        <style>
        /* 🌞 App background */
        .stApp {
            background: linear-gradient(to bottom right, #feffe6, #fff8e6);
        }

        /* 📚 Sidebar styling */
        [data-testid="stSidebar"] {
            background: linear-gradient(to bottom left, #f0e6ff, #d9cfff);
        }

        /* 🧠 Header styling */
        [data-testid="stHeader"] {
            background: linear-gradient(-225deg, #a6c6f7, #1667e0);
        }

        /* 🪴 Footer styling */
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: linear-gradient(to right, #3e3278, #b4aed6);
            color: #cbed8c;
            text-align: center;
            padding: 10px;
            font-size: 18px;
        }
        @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
        }
        div {
            animation: fadeIn 1s ease-in;
        }
        .footer {
            position: fixed;
            left: 0;
            bottom: 0;
            width: 100%;
            background: linear-gradient(to right, #3e3278, #b4aed6);
            color: #cbed8c;
            text-align: center;
            padding: 10px;
            font-size: 18px;
        }
        </style>
    """, unsafe_allow_html=True)
# 🌿 Styled header function
def styled_header(text):
    st.markdown(f"""
        <div style="background-color:#ffcc80; padding:15px; border-radius:10px; margin-bottom:20px;">
            <h1 style="color:#4e342e; text-align:center; font-family: 'Trebuchet MS'; font-size:28px;">
                {text}
            </h1>
        </div>
    """, unsafe_allow_html=True)

# 📌 Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# 🏠 Home Page
if app_mode == "Home":
    styled_header("🌿 PLANT DISEASE RECOGNITION SYSTEM 🌿")
    st.image("home_page.jpeg", use_container_width=True)

    st.markdown("""
        <div style="background-color:#e6ffe6; padding:20px; border-radius:12px; box-shadow:0 0 10px #b2dfdb;">
            <p style="font-size:18px; color:#2e7d32; font-family:'Segoe UI';">
                Welcome to the Plant Disease Recognition System! 🌿🔍<br><br>
                Upload an image of a plant, and our system will analyze it to detect any signs of diseases.<br><br>
                This tool is designed to help farmers, researchers, and plant enthusiasts identify issues early and take action. 🍀<br><br>
                Navigate through the sidebar to explore features like:<br>
                • 📸 Disease Recognition<br>
                • 📖 About the Project<br><br>
                Let's grow healthier plants together! 🌱✨
            </p>
        </div>
    """, unsafe_allow_html=True)

# 📖 About Page
elif app_mode == "About":
    styled_header("📖 ABOUT THIS PROJECT")
    st.markdown("""
        <div style="background-color:#fff3e0; padding:20px; border-radius:12px; box-shadow:0 0 10px #ffe0b2;">
            <p style="font-size:18px; color:#6d4c41; font-family:'Segoe UI';">
                This project uses deep learning to identify plant diseases from leaf images. 🧪<br><br>
                Built with TensorFlow, Keras, and Streamlit, it aims to provide actionable insights for agriculture.<br><br>
                Features include:<br>
                • 🌿 Image-based disease detection<br>
                • 🧪 Chemical and fertilizer recommendations<br>
                • 📊 Visual feedback and user-friendly interface<br><br>
                Developed by Shubham with a passion for blending science, design, and usability. 💡
            </p>
        </div>
    """, unsafe_allow_html=True)

# 🔍 Disease Recognition Page
elif app_mode == "Disease Recognition":
    styled_header("🔬 UPLOAD & DIAGNOSE PLANT DISEASE")
    st.markdown("""
        <div style="background-color:#e3f2fd; padding:20px; border-radius:12px; box-shadow:0 0 10px #90caf9;">
            <p style="font-size:18px; color:#1565c0; font-family:'Segoe UI';">
                Upload a clear image of a plant leaf below. 🌿<br><br>
                Our model will analyze the image and predict the disease type, if any.<br><br>
                You'll also receive tailored recommendations for treatment and fertilizer use. 🧪🌱
            </p>
        </div>
    """, unsafe_allow_html=True)

    test_image = st.file_uploader("📤 Upload Leaf Image", type=["jpg", "jpeg", "png"])
    if test_image:
        st.image(test_image, caption="Uploaded Leaf", use_container_width=True)
        st.toast("Image uploaded successfully! 🔍", icon="✅")
        # Add prediction logic here

    if st.button("🖼️ Show Image"):
        if test_image:
            st.image(test_image, use_container_width=True)

    if st.button("🔍 Predict Disease"):
        if test_image:
            st.toast("Prediction complete! 🌿", icon="🌟")

            result_index, confidence, model, input_arr = model_prediction(test_image)

            # Class labels
            class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                          'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                          'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                          'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                          'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                          'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                          'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                          'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                          'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                          'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                          'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                          'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                          'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                          'Tomato___healthy']

            predicted_label = class_name[result_index]
            st.success(f"Model predicts: **{predicted_label}**")
            st.markdown(f"<p style='color:#2e7d32; font-size:18px;'>🔍 Confidence: <strong>{confidence*100:.2f}%</strong></p>", unsafe_allow_html=True)

            # Chemical treatment recommendations
            # Fertilizer suggestions dictionary
            fertilizer_recommendations = {
                'Apple___Apple_scab': "Use nitrogen-rich fertilizer and apply fungicide sprays.",
                'Apple___Black_rot': "Apply balanced NPK and copper-based fungicide.",
                'Apple___Cedar_apple_rust': "Use sulfur-based fungicide and potassium-rich fertilizer.",
                'Apple___healthy': "Maintain balanced NPK and organic compost.",
                'Blueberry___healthy': "Use acidic fertilizer with ammonium sulfate.",
                'Cherry_(including_sour)___Powdery_mildew': "Apply sulfur fungicide and potassium nitrate.",
                'Cherry_(including_sour)___healthy': "Use compost and balanced micronutrients.",
                'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Apply nitrogen and foliar fungicide.",
                'Corn_(maize)___Common_rust_': "Use phosphorus-rich fertilizer and rust-resistant varieties.",
                'Corn_(maize)___Northern_Leaf_Blight': "Apply potassium and foliar sprays.",
                'Corn_(maize)___healthy': "Use urea and balanced micronutrients.",
                'Grape___Black_rot': "Apply copper fungicide and potassium sulfate.",
                'Grape___Esca_(Black_Measles)': "Use organic compost and avoid overwatering.",
                'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Apply foliar sprays and magnesium-rich fertilizer.",
                'Grape___healthy': "Use balanced NPK and organic mulch.",
                'Orange___Haunglongbing_(Citrus_greening)': "Apply zinc and manganese foliar sprays.",
                'Peach___Bacterial_spot': "Use copper fungicide and nitrogen-rich fertilizer.",
                'Peach___healthy': "Apply compost and calcium nitrate.",
                'Pepper,_bell___Bacterial_spot': "Use copper sprays and nitrogen fertilizer.",
                'Pepper,_bell___healthy': "Apply balanced NPK and organic compost.",
                'Potato___Early_blight': "Use phosphorus-rich fertilizer and chlorothalonil fungicide.",
                'Potato___Late_blight': "Apply metalaxyl fungicide and potassium nitrate.",
                'Potato___healthy': "Use compost and balanced NPK.",
                'Raspberry___healthy': "Apply compost and potassium sulfate.",
                'Soybean___healthy': "Use nitrogen-fixing inoculants and potassium fertilizer.",
                'Squash___Powdery_mildew': "Apply sulfur fungicide and compost tea.",
                'Strawberry___Leaf_scorch': "Use potassium-rich fertilizer and copper sprays.",
                'Strawberry___healthy': "Apply compost and micronutrients.",
                'Tomato___Bacterial_spot': "Use copper fungicide and nitrogen fertilizer.",
                'Tomato___Early_blight': "Apply phosphorus-rich fertilizer and fungicide.",
                'Tomato___Late_blight': "Use metalaxyl fungicide and potassium nitrate.",
                'Tomato___Leaf_Mold': "Apply calcium nitrate and improve airflow.",
                'Tomato___Septoria_leaf_spot': "Use potassium-rich fertilizer and chlorothalonil.",
                'Tomato___Spider_mites Two-spotted_spider_mite': "Apply neem oil and micronutrient mix.",
                'Tomato___Target_Spot': "Use magnesium-rich foliar spray and fungicide.",
                'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Apply seaweed extract and potassium sulfate.",
                'Tomato___Tomato_mosaic_virus': "Use compost and avoid nitrogen overdose.",
                'Tomato___healthy': "Maintain balanced NPK and regular composting."
            }
            predicted_disease = class_name[result_index]
            fertilizer_suggestion = fertilizer_recommendations.get(predicted_disease, "No recommendation available.")


            st.success(f"🌿 Model is predicting:\n {predicted_disease}\n\n💡 Fertilizer suggestion:\n{fertilizer_suggestion}")

                
            
            # Layer-wise activations
            st.markdown("## 🧠 Layer-wise Activation Viewer")
            show_layer_activations(model, input_arr)

# 📎 Footer
def show_footer():
    st.markdown("""
        <div class="footer">
            Made with 💚 by Shubham | Plant Health AI 🌿
        </div>
    """, unsafe_allow_html=True)
