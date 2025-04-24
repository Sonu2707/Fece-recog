import base64
import traceback
from datetime import datetime
from io import BytesIO
from pathlib import Path

import streamlit as st
from PIL import Image
import numpy as np
import cv2
import plotly.express as px
from fpdf import FPDF
import plotly.graph_objects as go

# Enhanced error handling with version checks
try:
    import tensorflow as tf
    tf_version = tf.__version__
    if tf_version != '2.11.0':
        st.warning(f"TensorFlow version {tf_version} detected (expected 2.11.0)")
    
    from deepface import DeepFace
    from deepface.commons import functions
except ImportError as e:
    st.error(f"Critical import failed: {str(e)}")
    st.error("Please check the requirements.txt and install dependencies")
    with st.expander("Debug Info"):
        st.code(traceback.format_exc())
    st.stop()

# App configuration
st.set_page_config(
    page_title="FaceX AI Analyzer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injection with dark/light mode support
def inject_css():
    st.markdown("""
    <style>
        :root {
            --primary-color: #4f8bf9;
            --background-color: #ffffff;
            --secondary-background: #f0f2f6;
            --text-color: #31333F;
            --font: "Source Sans Pro", sans-serif;
        }
        
        @media (prefers-color-scheme: dark) {
            :root {
                --background-color: #0e1117;
                --secondary-background: #1e2229;
                --text-color: #f0f2f6;
            }
        }
        
        .card {
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
            background: var(--secondary-background);
            transition: transform 0.2s;
            color: var(--text-color);
        }
        .gallery-img {
            border-radius: 10px;
            transition: transform 0.3s ease;
            cursor: pointer;
            max-width: 100%;
        }
        .stButton>button {
            border-radius: 8px;
            padding: 8px 16px;
            background-color: var(--primary-color);
            color: white;
        }
        .report-section {
            border-left: 4px solid var(--primary-color);
            padding-left: 1rem;
        }
        [data-testid="stExpander"] {
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: 8px;
        }
        .analysis-badge {
            display: inline-block;
            padding: 3px 8px;
            border-radius: 12px;
            background-color: var(--primary-color);
            color: white;
            font-size: 0.8em;
            margin-right: 5px;
        }
    </style>
    """, unsafe_allow_html=True)

inject_css()

# Session state initialization
if 'uploaded_images' not in st.session_state:
    st.session_state.uploaded_images = []

# Image processing functions
def analyze_faces(img_path):
    try:
        results = DeepFace.analyze(
            img_path=img_path,
            actions=['age', 'gender', 'emotion', 'race'],
            detector_backend='opencv',
            enforce_detection=False,  # Changed to handle cases with no faces
            silent=True
        )
        return results
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        return None

# PDF Report Generator with enhanced styling
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'FaceX AI Analysis Report', 0, 1, 'C')
        self.ln(10)
    
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
    
    def add_image_analysis(self, img_path, analysis, timestamp):
        self.add_page()
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, f"Analysis for {timestamp}", 0, 1)
        self.ln(5)
        
        # Add image with border
        self.image(img_path, w=80, h=60)
        self.ln(10)
        
        # Analysis results
        self.set_font('Arial', '', 10)
        self.cell(40, 6, "Age:", 0, 0)
        self.cell(0, 6, str(analysis['age']), 0, 1)
        
        self.cell(40, 6, "Gender:", 0, 0)
        self.cell(0, 6, analysis['gender'], 0, 1)
        
        self.cell(40, 6, "Dominant Emotion:", 0, 0)
        self.cell(0, 6, analysis['dominant_emotion'], 0, 1)
        
        self.cell(40, 6, "Dominant Race:", 0, 0)
        self.cell(0, 6, analysis['dominant_race'], 0, 1)
        
        # Emotion distribution chart
        self.ln(10)
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, "Emotion Distribution:", 0, 1)
        
        # Create simple bar chart
        emotions = analysis['emotion']
        max_val = max(emotions.values())
        
        for emotion, value in emotions.items():
            self.set_font('Arial', '', 8)
            self.cell(30, 6, emotion, 0, 0)
            self.set_fill_color(79, 139, 249)
            self.cell(0, 6, '', 'LRTB', 1, 'L', fill=True, w=value/max_val*100)

# UI Components
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload", "Gallery", "Analysis", "Reports"])
    
    # Image upload handling
    if page == "Upload":
        st.title("📤 Image Upload")
        with st.expander("ℹ️ Instructions"):
            st.markdown("""
            - Upload clear facial images (JPG/PNG)
            - Multiple images can be uploaded at once
            - Images will appear in the Gallery tab
            """)
            
        uploaded_files = st.file_uploader(
            "Choose files", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="uploader"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                try:
                    img = Image.open(uploaded_file)
                    img_data = {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image": img,
                        "analysis": None,
                        "filename": uploaded_file.name,
                        "temp_path": None
                    }
                    st.session_state.uploaded_images.append(img_data)
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {str(e)}")
            
            st.toast(f"Added {len(uploaded_files)} images to gallery!", icon="✅")

    # Image gallery view
    elif page == "Gallery":
        st.title("🖼 Image Gallery")
        if not st.session_state.uploaded_images:
            st.info("No images uploaded yet! Use the Upload tab to add images.")
            return
            
        cols = st.columns(3)
        for idx, img_data in enumerate(st.session_state.uploaded_images):
            with cols[idx % 3]:
                with st.container():
                    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                    
                    # Display image with analysis status
                    st.image(img_data["image"], use_column_width=True, 
                           caption=f"{img_data['filename']} - {img_data['timestamp']}")
                    
                    # Analysis button and status
                    if img_data["analysis"]:
                        st.markdown("<span class='analysis-badge'>Analyzed</span>", 
                                  unsafe_allow_html=True)
                    else:
                        if st.button(f"Analyze #{idx+1}", key=f"analyze_{idx}"):
                            with st.spinner("Analyzing facial features..."):
                                try:
                                    img_path = f"temp_{idx}.jpg"
                                    img_data["image"].save(img_path)
                                    analysis = analyze_faces(img_path)
                                    if analysis:
                                        img_data["analysis"] = analysis[0]
                                        img_data["temp_path"] = img_path
                                        st.toast("Analysis complete!", icon="✅")
                                        st.experimental_rerun()
                                except Exception as e:
                                    st.error(f"Analysis failed: {str(e)}")
                    
                    st.markdown("</div>", unsafe_allow_html=True)

    # Analysis dashboard
    elif page == "Analysis":
        st.title("📊 Analysis Dashboard")
        if not st.session_state.uploaded_images:
            st.info("Upload images first from the Upload tab!")
            return
            
        selected_idx = st.sidebar.selectbox(
            "Select Image",
            options=[f"Image {i+1} - {img['filename']}" 
                    for i, img in enumerate(st.session_state.uploaded_images)]
        )
        idx = int(selected_idx.split()[1]) - 1
        img_data = st.session_state.uploaded_images[idx]
        
        if not img_data["analysis"]:
            st.warning("This image hasn't been analyzed yet. Please run analysis from the Gallery tab.")
            return
            
        analysis = img_data["analysis"]
        
        # Main analysis display
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("Image Preview")
            st.image(img_data["image"], use_column_width=True)
            
            # Basic info card
            st.markdown("### Analysis Summary")
            st.markdown(f"""
            <div class='card'>
                <p><strong>Age:</strong> {analysis['age']}</p>
                <p><strong>Gender:</strong> {analysis['gender']}</p>
                <p><strong>Dominant Emotion:</strong> {analysis['dominant_emotion']}</p>
                <p><strong>Dominant Race:</strong> {analysis['dominant_race']}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.subheader("Detailed Analysis")
            
            # Emotion distribution
            st.markdown("#### Emotion Distribution")
            emotions = analysis['emotion']
            fig = px.bar(
                x=list(emotions.keys()),
                y=list(emotions.values()),
                color=list(emotions.keys()),
                labels={'x': 'Emotion', 'y': 'Confidence'},
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Race distribution
            if 'race' in analysis:
                st.markdown("#### Race Distribution")
                race_data = analysis['race']
                fig = px.pie(
                    names=list(race_data.keys()),
                    values=list(race_data.values()),
                    hole=0.4,
                    color_discrete_sequence=px.colors.qualitative.Pastel
                )
                st.plotly_chart(fig, use_container_width=True)

    # Report generation
    elif page == "Reports":
        st.title("📄 Generate Reports")
        if not st.session_state.uploaded_images:
            st.info("No images available for reporting. Upload images first.")
            return
            
        if st.button("📥 Generate Full Report PDF"):
            try:
                with st.spinner("Generating report..."):
                    pdf = PDFReport()
                    
                    for img_data in st.session_state.uploaded_images:
                        if img_data["analysis"]:
                            # Use temp path if available, otherwise create temp file
                            if img_data.get("temp_path"):
                                img_path = img_data["temp_path"]
                            else:
                                img_path = f"report_temp_{id(img_data)}.jpg"
                                img_data["image"].save(img_path)
                            
                            pdf.add_image_analysis(
                                img_path,
                                img_data["analysis"],
                                img_data["timestamp"]
                            )
                            
                            # Clean up if we created a temp file just for reporting
                            if not img_data.get("temp_path"):
                                Path(img_path).unlink()
                    
                    report_bytes = pdf.output(dest='S').encode('latin-1')
                    
                    # Download button
                    st.download_button(
                        label="⬇️ Download PDF Report",
                        data=report_bytes,
                        file_name="facex_report.pdf",
                        mime="application/pdf",
                        type="primary"
                    )
            except Exception as e:
                st.error(f"Report generation failed: {str(e)}")
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())

    # Debug section
    with st.expander("ℹ️ Debug Console"):
        st.code(f"Session state keys: {list(st.session_state.keys())}")
        if st.button("Clear Cache"):
            # Clean up temp files
            for img_data in st.session_state.uploaded_images:
                if img_data.get("temp_path") and Path(img_data["temp_path"]).exists():
                    Path(img_data["temp_path"]).unlink()
            st.session_state.clear()
            st.experimental_rerun()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        st.error(f"Critical error: {str(e)}")
        with st.expander("Error Details"):
            st.code(traceback.format_exc())
        st.stop()
