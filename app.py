# app.py
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

# Error handling for critical imports
try:
    from deepface import DeepFace
    from deepface.commons import functions
except ImportError as e:
    st.error(f"Critical import failed: {str(e)}")
    st.stop()

# App configuration
st.set_page_config(
    page_title="FaceX AI Analyzer",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS injection
def inject_css():
    st.markdown("""
    <style>
        .card {
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin: 10px 0;
            background: var(--background-color);
            transition: transform 0.2s;
        }
        .card:hover {
            transform: translateY(-2px);
        }
        .gallery-img {
            border-radius: 10px;
            transition: transform 0.3s ease;
            cursor: pointer;
        }
        .gallery-img:hover {
            transform: scale(1.03);
        }
        .stButton>button {
            border-radius: 8px;
            padding: 8px 16px;
        }
        .report-section {
            border-left: 4px solid #4CAF50;
            padding-left: 1rem;
        }
        [data-testid="stExpander"] {
            border: 1px solid rgba(0,0,0,0.1);
            border-radius: 8px;
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
            enforce_detection=True,
            silent=True
        )
        return results
    except Exception as e:
        st.error(f"Analysis failed: {str(e)}")
        return None

def verify_faces(img1_path, img2_path):
    try:
        result = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name='VGG-Face',
            detector_backend='opencv',
            distance_metric='cosine'
        )
        return result
    except Exception as e:
        st.error(f"Verification failed: {str(e)}")
        return None

# PDF Report Generator
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'FaceX AI Analysis Report', 0, 1, 'C')
    
    def add_image_analysis(self, img_path, analysis, timestamp):
        self.add_page()
        self.set_font('Arial', '', 10)
        self.image(img_path, w=80)
        self.ln(10)
        self.cell(0, 10, f"Analyzed at: {timestamp}", ln=1)
        self.multi_cell(0, 6, f"Age: {analysis['age']}\nGender: {analysis['gender']}\nDominant Emotion: {analysis['dominant_emotion']}\nRace: {analysis['dominant_race']}")

# UI Components
def main():
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Upload", "Gallery", "Analysis", "Reports"])
    
    # Image upload handling
    if page == "Upload":
        st.title("📤 Image Upload")
        uploaded_files = st.file_uploader(
            "Upload facial images (JPG, PNG)", 
            type=['jpg', 'jpeg', 'png'],
            accept_multiple_files=True,
            key="uploader"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                img = Image.open(uploaded_file)
                img_data = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "image": img,
                    "analysis": None,
                    "filename": uploaded_file.name
                }
                st.session_state.uploaded_images.append(img_data)
            st.success(f"Added {len(uploaded_files)} images to gallery!")

    # Image gallery view
    elif page == "Gallery":
        st.title("🖼 Image Gallery")
        if not st.session_state.uploaded_images:
            st.info("No images uploaded yet!")
            return
            
        cols = st.columns(3)
        for idx, img_data in enumerate(st.session_state.uploaded_images):
            with cols[idx % 3]:
                with st.container():
                    st.markdown(f"<div class='card'>", unsafe_allow_html=True)
                    st.image(img_data["image"], use_column_width=True, caption=img_data["filename"])
                    st.caption(f"Uploaded: {img_data['timestamp']}")
                    if st.button(f"Analyze #{idx+1}", key=f"analyze_{idx}"):
                        with st.spinner("Analyzing facial features..."):
                            img_path = f"temp_{idx}.jpg"
                            img_data["image"].save(img_path)
                            analysis = analyze_faces(img_path)
                            if analysis:
                                img_data["analysis"] = analysis[0]
                                st.success("Analysis complete!")
                    st.markdown("</div>", unsafe_allow_html=True)

    # Analysis dashboard
    elif page == "Analysis":
        st.title("📊 Analysis Dashboard")
        if not st.session_state.uploaded_images:
            st.info("Upload images first!")
            return
            
        selected_idx = st.sidebar.selectbox(
            "Select Image",
            options=[f"Image {i+1}" for i in range(len(st.session_state.uploaded_images))]
        )
        idx = int(selected_idx.split()[-1]) - 1
        img_data = st.session_state.uploaded_images[idx]
        
        if not img_data["analysis"]:
            st.warning("Run analysis from Gallery first!")
            return
            
        analysis = img_data["analysis"]
        
        # Visualization
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Demographics")
            fig = px.pie(
                names=[f"Age {analysis['age']}", analysis['gender']],
                values=[1, 1],
                hole=0.4,
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Emotion Distribution")
            emotions = analysis['emotion']
            fig = px.bar(
                x=list(emotions.keys()),
                y=list(emotions.values()),
                color=list(emotions.keys()),
                labels={'x': 'Emotion', 'y': 'Confidence'},
                color_discrete_sequence=px.colors.sequential.Plasma
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Reverse image search
        st.subheader("Reverse Image Search")
        img_bytes = BytesIO()
        img_data["image"].save(img_bytes, format='PNG')
        b64_img = base64.b64encode(img_bytes.getvalue()).decode()
        
        search_engines = {
            "Google": f"https://www.google.com/searchbyimage?image_url=data:image/png;base64,{b64_img}",
            "Bing": f"https://www.bing.com/images/search?q=imgurl:{b64_img}",
            "Yahoo": f"https://images.search.yahoo.com/search/images?imgurl=data:image/png;base64,{b64_img}",
            "DuckDuckGo": f"https://duckduckgo.com/?q={b64_img}&iax=images&ia=images"
        }
        
        cols = st.columns(4)
        for idx, (engine, url) in enumerate(search_engines.items()):
            cols[idx].markdown(
                f"<a href='{url}' target='_blank' class='card' style='display: block; text-align: center; padding: 10px;'>{engine}</a>",
                unsafe_allow_html=True
            )

    # Report generation
    elif page == "Reports":
        st.title("📄 Generate Reports")
        if st.button("📥 Download Full Report"):
            pdf = PDFReport()
            for img_data in st.session_state.uploaded_images:
                if img_data["analysis"]:
                    img_path = f"report_temp_{id(img_data)}.jpg"
                    img_data["image"].save(img_path)
                    pdf.add_image_analysis(
                        img_path,
                        img_data["analysis"],
                        img_data["timestamp"]
                    )
                    Path(img_path).unlink()
            
            report_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button(
                label="⬇️ Download PDF Report",
                data=report_bytes,
                file_name="facex_report.pdf",
                mime="application/pdf"
            )

    # Debug section
    with st.expander("Debug Console"):
        st.code(f"Session state: {st.session_state}")
        if st.button("Clear Cache"):
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
