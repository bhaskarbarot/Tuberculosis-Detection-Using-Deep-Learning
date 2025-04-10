import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
import os
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
import time
from PIL import Image
import io
import pandas as pd
import datetime
from pathlib import Path
import requests
import tempfile

st.set_page_config(page_title="TB Detection System", layout="wide", page_icon="🫁")

IMG_SIZE = 224

@st.cache_resource
def load_saved_model(model_path):
    return load_model(model_path)

def preprocess_image(image):
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    
    return image

def predict_tb(image, model):
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    prob = float(prediction[0][0])
    class_idx = 1 if prob > 0.5 else 0
    class_name = "Tuberculosis" if class_idx == 1 else "Normal"
    
    return class_name, prob

# New function that uses the actual image links
@st.cache_data
def get_image_from_drive(file_id):
    # Convert Google Drive sharing link to direct download link
    direct_link = f"https://drive.google.com/uc?export=download&id={file_id}"
    
    try:
        response = requests.get(direct_link)
        image = Image.open(io.BytesIO(response.content))
        return image
    except Exception as e:
        st.error(f"Error loading image: {e}")
        # Return a fallback image if there's an error
        return Image.new('RGB', (600, 400), color=(200, 200, 200))

def get_placeholder_image(label):
    # Dictionary mapping labels to file IDs from the Google Drive links
    image_ids = {
        "vgg16": "1s3PeG3y0Tz_hPFjoY9f1DcFjk92XQ8tR",
        "xray_analysis": "1icdCMnUB0eotB4E_nzMsyUeZCTwxjKiM",
        "tb_spread": "1Dh-rVYQaYBbW6ig2NtReKsAUJqJtzjuo",
        "tb_prevention": "1Fq6feYtzboM9Rl480ljdBiv9Fi-1xh3x",
        "tb_treatment": "1cnQRPJOQ19uq4RU07qCZ9qKvV6gPLjo8",
        "tb_global": "14dPTgBtjSc9vaFIgssPlkiIELH616xGv",
        "tb_bacteria": "14qx04pL7oN7mZVC158GxvQIwACsCUhGh",
        "login_hero": "1m7UBfRXvcBvEqu2XcqLkST3EUNHdw3D-",
        # Adding the new images
        "normal_xray": "17vqMwcPg41LF9Q2HKdJxRONABhx9GQOS",
        "tb_xray": "1W68OJJh1jSGEAHiTGLagwixoDP72mmH6"
    }
    
    # For the other placeholder images we'll keep generating them as before
    if label in image_ids:
        return get_image_from_drive(image_ids[label])
    else:
        # Generate dummy images for labels not in our dictionary
        return generate_dummy_image(600, 400, "gray", f"{label.replace('_', ' ').title()}")

def generate_dummy_image(width, height, color_code, text):
    img = np.ones((height, width, 3), dtype=np.uint8)
    
    if color_code == "red":
        img[:] = (200, 100, 100)  # BGR for reddish
    elif color_code == "green":
        img[:] = (100, 200, 100)  # BGR for greenish
    elif color_code == "blue":
        img[:] = (100, 100, 200)  # BGR for bluish
    elif color_code == "orange":
        img[:] = (100, 150, 240)  # BGR for orangish
    elif color_code == "purple":
        img[:] = (180, 100, 180)  # BGR for purplish
    else:
        img[:] = (200, 200, 200)  # BGR for grayish
    
    thickness = 2
    font_scale = 0.8
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    cv2.putText(img, text, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def display_class_distribution():
    st.subheader("Dataset Class Distribution")
    
    normal_count = 514
    tb_count = 2494
    
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=['Normal', 'TB'], y=[normal_count, tb_count], palette=['skyblue', 'lightcoral'], ax=ax)
    ax.set_title('Class Distribution in Training Dataset')
    ax.set_ylabel('Count')
    total = normal_count + tb_count
    normal_percent = (normal_count / total) * 100
    tb_percent = (tb_count / total) * 100
    
    ax.text(0, normal_count/2, f"{normal_count} ({normal_percent:.1f}%)", ha='center', va='center', fontweight='bold')
    ax.text(1, tb_count/2, f"{tb_count} ({tb_percent:.1f}%)", ha='center', va='center', fontweight='bold')
    
    plt.tight_layout()
    st.pyplot(fig)

def display_tb_symptoms():
    st.subheader("Common TB Symptoms")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("""
        ### Primary Symptoms:
        - Persistent cough (lasting more than 3 weeks)
        - Coughing up blood or mucus
        - Chest pain during breathing or coughing
        - Unintentional weight loss
        - Fatigue and weakness
        - Fever and night sweats
        - Loss of appetite
        
        ### Additional Symptoms:
        - Shortness of breath
        - Chills
        - Pain while breathing
        """)
        
        st.warning("**Note:** TB can remain dormant without symptoms (latent TB). Only active TB causes symptoms and can spread to others.")
        
    with col2:
        st.image(get_placeholder_image("tb_bacteria"), caption="Tuberculosis in lungs", use_container_width=True)

def display_tb_prevention():
    st.subheader("TB Prevention & Management")
    
    cols = st.columns(3)
    
    with cols[0]:
        st.markdown("### Prevention")
        st.image(get_placeholder_image("tb_prevention"), caption="TB Prevention", use_container_width=True)
        st.markdown("""
        - BCG vaccination for children
        - Screening high-risk populations
        - Proper ventilation in public spaces
        - Early testing and diagnosis
        - Covering mouth when coughing/sneezing
        """)
        
    with cols[1]:
        st.markdown("### Treatment")
        st.image(get_placeholder_image("tb_treatment"), caption="TB Medication", use_container_width=True)
        st.markdown("""
        - 6-9 month antibiotic regimen
        - Combination of multiple drugs
        - Directly Observed Therapy (DOT)
        - Regular follow-up testing
        - Complete the full course of treatment
        """)
        
    with cols[2]:
        st.markdown("### Global Efforts")
        st.image(get_placeholder_image("tb_global"), caption="TB Control Programs", use_container_width=True)
        st.markdown("""
        - WHO End TB Strategy
        - Universal healthcare coverage
        - Research for new diagnostics
        - Development of new TB vaccines
        - Community awareness programs
        """)

def display_model_overview():
    st.subheader("About Our AI Detection System")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### VGG16 Deep Learning Model
        
        Our system uses a state-of-the-art VGG16 convolutional neural network to analyze chest X-rays for TB detection.
        
        #### Key Features:
        - **High Accuracy**: 98% accuracy in detecting TB patterns
        - **Fast Analysis**: Results in seconds, not hours
        - **Detailed Confidence Scores**: Provides probability metrics
        - **Professional-Grade**: Built on medical imaging expertise
        """)
        
    with col2:
        st.image(get_placeholder_image("vgg16"), caption="VGG16 Architecture", use_container_width=True)

def initialize_user_db():
    # Use session state to store user data instead of file system
    if 'users_df' not in st.session_state:
        st.session_state.users_df = pd.DataFrame(columns=['email', 'registration_date', 'last_login'])
    return st.session_state.users_df

def user_exists(email, users_df):
    # Check if email column exists before checking if the email is in it
    return 'email' in users_df.columns and email in users_df['email'].values

def register_user(email, users_df):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    new_user = pd.DataFrame({
        'email': [email],
        'registration_date': [now],
        'last_login': [now]
    })
    users_df = pd.concat([users_df, new_user], ignore_index=True)
    # Store in session state
    st.session_state.users_df = users_df
    return users_df

def update_last_login(email, users_df):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    users_df.loc[users_df['email'] == email, 'last_login'] = now
    # Store in session state
    st.session_state.users_df = users_df

def display_user_count(users_df):
    st.subheader("User Statistics")
    num_users = len(users_df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Registered Users", num_users)
    with col2:
        # Count new users today - handle potential errors if date format is incorrect
        try:
            new_users_today = sum(pd.to_datetime(users_df['registration_date']).dt.date == datetime.datetime.now().date())
        except:
            new_users_today = 0
        st.metric("New Users Today", new_users_today)

def login_page():
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
        st.session_state.email = None

    # Initialize CSS styles
    st.markdown("""
    <style>
    .login-container {
        max-width: 600px;
        margin: 0 auto;
        padding: 30px;
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    .login-header {
        text-align: center;
        margin-bottom: 30px;
    }
    .login-header h1 {
        color: #1E88E5;
        font-size: 36px;
        font-weight: bold;
    }
    .login-header p {
        color: #546E7A;
        font-size: 18px;
        margin-top: 10px;
    }
    .login-footer {
        text-align: center;
        margin-top: 20px;
        color: #78909C;
        font-size: 12px;
    }
    .divider {
        text-align: center;
        margin: 20px 0;
        border-bottom: 1px solid #e0e0e0;
        line-height: 0;
    }
    .divider span {
        background-color: #f8f9fa;
        padding: 0 10px;
        color: #78909C;
    }
    .features {
        display: flex;
        justify-content: space-around;
        margin: 30px 0;
        text-align: center;
    }
    .feature {
        flex: 1;
        padding: 15px;
        margin: 0 10px;
        background-color: #E3F2FD;
        border-radius: 5px;
    }
    .feature h3 {
        color: #1565C0;
        font-size: 16px;
        margin-bottom: 5px;
    }
    .feature p {
        color: #546E7A;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

    # Create columns for layout
    col1, col2 = st.columns([2, 1])

    with col1:
        # Login container
        st.markdown("""
        <div class="login-container">
            <div class="login-header">
                <h1>TB Detection System</h1>
                <p>AI-powered tuberculosis detection from chest X-rays</p>
            </div>
        """, unsafe_allow_html=True)

        # Form for login/registration
        users_df = initialize_user_db()
        
        email = st.text_input("Email Address", placeholder="Enter your email to continue", key="login_email")
        
        login_button = st.button("Continue", type="primary", use_container_width=True)
        
        if login_button:
            if not email or "@" not in email:
                st.error("Please enter a valid email address.")
            else:
                if user_exists(email, users_df):
                    update_last_login(email, users_df)
                    st.success(f"Welcome back! You are now signed in.")
                else:
                    users_df = register_user(email, users_df)
                    st.success(f"Account created successfully! Welcome to TB Detection System.")
                
                # Set session state
                st.session_state.logged_in = True
                st.session_state.email = email
                
                # Using a JavaScript hack to force page reload instead of st.page_link
                st.markdown("""
                <script>
                    setTimeout(function(){
                        window.location.reload();
                    }, 1500);
                </script>
                """, unsafe_allow_html=True)
        
        # Features section
        st.markdown("""
            <div class="divider"><span>Key Features</span></div>
            <div class="features">
                <div class="feature">
                    <h3>AI Detection</h3>
                    <p>98% accurate TB detection using VGG16 neural network</p>
                </div>
                <div class="feature">
                    <h3>Fast Results</h3>
                    <p>Get analysis results in seconds, not hours</p>
                </div>
                <div class="feature">
                    <h3>Educational</h3>
                    <p>Learn about TB symptoms, prevention & treatment</p>
                </div>
            </div>
            <div class="login-footer">
                Created by Bhaskar Barot
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        # Display login hero image
        st.image(get_placeholder_image("login_hero"), use_container_width=True)
        
        # Info box
        st.info("""
        **About This System**
        
        This AI-powered tool assists healthcare professionals in detecting potential tuberculosis cases from chest X-ray images.
        
        This system is for demonstration and educational purposes only.
        """)

def main():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        login_page()
        return
        
    st.title("Tuberculosis Detection from Chest X-rays")
    st.markdown("""
    <style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 26px;
        font-weight: 600;
        color: #0277BD;
        margin-top: 30px;
    }
    .important-text {
        background-color: #E3F2FD;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #1E88E5;
    }
    .stButton>button {
        background-color: #1E88E5;
        color: white;
        font-weight: bold;
        padding: 10px 20px;
        border-radius: 5px;
    }
    </style>
    
    <div class="main-header">AI-Powered TB Detection System</div>
    """, unsafe_allow_html=True)
    
    # Use the session state for user data
    if 'users_df' not in st.session_state:
        st.session_state.users_df = pd.DataFrame(columns=['email', 'registration_date', 'last_login'])
    users_df = st.session_state.users_df
    
    st.sidebar.title(f"Welcome, {st.session_state.email.split('@')[0]}")
    
    if st.sidebar.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.email = None
        st.rerun()
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("", ["Home", "Upload & Predict"])
    
    # Try to get the model path from file or use a default
    model_path = "/home/ec2-user/TB_project/vgg16_best_model.h5"
    
    model = None
    try:
        model = load_saved_model(model_path)
        st.sidebar.success(f"✅ VGG16 model loaded successfully!")
    except Exception as e:
        st.sidebar.error(f"❌ Error loading model: {e}")
        st.sidebar.info(f"Please ensure the model exists at: {model_path}")
        
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### Resources
    - [WHO TB Information](https://www.who.int/health-topics/tuberculosis)
    - [CDC TB Facts](https://www.cdc.gov/tb/default.htm)
    - [TB Prevention Guide](https://www.cdc.gov/tb/topic/basics/tbprevention.htm)
    
    ### Indian TB Resources
    - [National TB Elimination Programme (NTEP)](https://tbcindia.gov.in/)
    - [Ministry of Health and Family Welfare - TB](https://main.mohfw.gov.in/Major-Programmes/non-communicable-diseases-injury-trauma/Non-Communicable-Disease-II/National-Tuberculosis-Elimination-Programme)
    - [TB Facts India](https://tbfacts.org/tb-india/)
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    <div style="text-align: center; margin-top: 20px; color: #78909C; font-size: 12px;">
        Created by Bhaskar Barot
    </div>
    """, unsafe_allow_html=True)
    
    if page == "Home":
        display_user_count(users_df)
        
        st.header("Understanding Tuberculosis")
        
        st.markdown("""
        <div class="important-text">
        Tuberculosis (TB) is an infectious disease that primarily affects the lungs and is caused by <i>Mycobacterium tuberculosis</i> bacteria.
        It remains one of the world's deadliest infectious diseases, with over <b>10 million new cases</b> and approximately <b>1.5 million deaths</b> annually.
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Global TB Impact")
        cols = st.columns(4)
        
        with cols[0]:
            st.metric("Annual Cases", "10+ million", "+3%")
        with cols[1]:
            st.metric("Annual Deaths", "1.5 million", "-2%")
        with cols[2]:
            st.metric("Drug Resistant Cases", "500,000+", "+5%")
        with cols[3]:
            st.metric("Children Affected", "1.1 million", "+1%")
        
        display_tb_symptoms()
        
        display_tb_prevention()
        
        st.subheader("How TB Spreads")
        st.image(get_placeholder_image("tb_spread"), caption="TB Transmission", use_container_width=True)
        
        display_model_overview()
        
        display_class_distribution()
        
        st.markdown("""
        <div class="important-text">
        <h3 style="text-align: center;">Ready to analyze a chest X-ray?</h3>
        <p style="text-align: center;">Navigate to 'Upload & Predict' in the sidebar to upload and analyze an X-ray image.</p>
        </div>
        """, unsafe_allow_html=True)
    
    elif page == "Upload & Predict":
        st.header("Upload Chest X-ray for TB Analysis")
        
        st.markdown("""
        <div class="important-text">
        <h4>Instructions:</h4>
        <ol>
            <li>Upload a chest X-ray image (JPG, PNG, or JPEG format)</li>
            <li>Click "Analyze X-ray" to get predictions</li>
            <li>Review the diagnosis and confidence score</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("See Example X-rays"):
            col1, col2 = st.columns(2)
            with col1:
                st.image(get_placeholder_image("normal_xray"), caption="Normal X-ray", use_container_width=True)
            with col2:
                st.image(get_placeholder_image("tb_xray"), caption="TB-Positive X-ray", use_container_width=True)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                
                st.image(image, caption="Uploaded X-ray Image", use_container_width=True)
                
                if st.button("Analyze X-ray"):
                    if model is not None:
                        with st.spinner("Analyzing image... Please wait"):
                            progress_bar = st.progress(0)
                            for i in range(100):
                                time.sleep(0.01)
                                progress_bar.progress(i + 1)
                            
                            class_name, probability = predict_tb(image, model)
                        
                        progress_bar.empty()
                        
                        st.success("Analysis complete!")
                        
                        with col2:
                            st.subheader("Diagnosis Result")
                            
                            if class_name == "Tuberculosis":
                                st.markdown(f"""
                                <div style="background-color: #FFEBEE; padding: 20px; border-radius: 10px; border-left: 5px solid #C62828;">
                                    <h3 style="color: #C62828;">Result: {class_name}</h3>
                                    <p><b>Confidence:</b> {probability*100:.2f}%</p>
                                    <p>The model has detected patterns consistent with <b>tuberculosis</b> in this X-ray.</p>
                                    <p><i>Note: This is an AI-assisted prediction and should be confirmed by a qualified healthcare professional.</i></p>
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div style="background-color: #E8F5E9; padding: 20px; border-radius: 10px; border-left: 5px solid #2E7D32;">
                                    <h3 style="color: #2E7D32;">Result: {class_name}</h3>
                                    <p><b>Confidence:</b> {probability*100:.2f}%</p>
                                    <p>The model does <b>not</b> detect patterns consistent with tuberculosis in this X-ray.</p>
                                    <p><i>Note: This is an AI-assisted prediction and should be confirmed by a qualified healthcare professional.</i></p>
                                </div>
                                """, unsafe_allow_html=True)
                            
                            fig, ax = plt.subplots(figsize=(4, 4))
                            ax.set_aspect('equal')
                            
                            if class_name == "Tuberculosis":
                                color = '#C62828'
                            else:
                                color = '#2E7D32'
                                
                            ax.pie([probability, 1-probability], 
                                  colors=[color, 'lightgray'],
                                  startangle=90, counterclock=False,
                                  wedgeprops=dict(width=0.3))
                            
                            ax.text(0, 0, f"{probability*100:.1f}%", 
                                   ha='center', va='center', fontsize=24)
                            
                            ax.set_title("Prediction Confidence")
                            st.pyplot(fig)
                            
                            if class_name == "Tuberculosis":
                                st.markdown("""
                                ### Recommended Next Steps:
                                1. Consult with a pulmonologist or TB specialist
                                2. Consider sputum test for confirmation
                                3. Follow up with additional imaging if needed
                                4. Discuss treatment options if confirmed
                                """)
                    else:
                        st.error("Model not loaded. Please check if the model file exists at the specified path.")
        
        if not uploaded_file:
            with col2:
                st.info("Upload an X-ray image to see the analysis results here.")
                st.image(get_placeholder_image("xray_analysis"), caption="Example of chest X-ray analysis", use_container_width=True)

if __name__ == "__main__":
    main()