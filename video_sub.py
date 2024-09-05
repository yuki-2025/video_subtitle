import streamlit as st
import tempfile
import os
from io import StringIO
import torch 
import numpy as np
import logging
import datetime
import time 
import psutil
from audio_utils import format_timestamp, generate_srt, transcribe_audio, set_logger 

# Configure logging 
class StreamlitHandler(logging.Handler):
    def __init__(self, placeholder):
        super().__init__()
        self.placeholder = placeholder
        self.log_output = StringIO()

    def emit(self, record):
        log_entry = f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {self.format(record)}"
        self.log_output.write(log_entry + '\n')
        self.placeholder.code(self.log_output.getvalue())

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 

def get_gpu_info():
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_properties(0)
        return f"GPU: {gpu.name}, Total Memory: {gpu.total_memory / 1e9:.2f} GB"
    return "GPU: Not available"

def get_cpu_info():
    cpu_info = psutil.cpu_freq()
    cpu_count = psutil.cpu_count(logical=False)
    cpu_logical_count = psutil.cpu_count(logical=True)
    return f"CPU: {cpu_count} physical cores, {cpu_logical_count} logical cores, Max Frequency: {cpu_info.max:.2f} MHz"
 
def main():
    st.set_page_config(page_title="Video Subtitle Generator", page_icon="🎬")
    st.markdown("""
    <style>
     
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
        border-radius: 5px;
        border: none;
    }
    </style>
    """, unsafe_allow_html=True)
        
    st.title("Video Subtitle Generator")
    st.markdown("Generate subtitles from an audio/video file, using OpenAI's Whisper model.")

    # Input section
    st.header("Input")
    
    with st.form(key='subtitle_form'):
    # Create two columns with 2:1 ratio
        col1, col2 = st.columns([2, 1])

        # Elements in the wider column (2/3 width)
        with col1:
            uploaded_file = st.file_uploader("Upload video/audio file", type=["mp3", "wav", "mp4"])

        # Elements in the narrower column (1/3 width)
        with col2:
            model_name = st.selectbox("Choose Whisper model", [
                "openai/whisper-base",
                "openai/whisper-tiny",
                "openai/whisper-small",
                "openai/whisper-medium",
                "openai/whisper-large"
            ])
            language = st.selectbox("Choose language", ["en", "fr", "de", "es", "it", "ja", "ko", "pt", "ru", "zh"])

        # Add a submit button to the form
        submit_button = st.form_submit_button(label='Generate Subtitles')
     
    st.subheader("Logs")
    # Create a placeholder for logs
    logs_placeholder = st.empty()

    # Add StreamlitHandler to logger
    streamlit_handler = StreamlitHandler(logs_placeholder)
    logger.addHandler(streamlit_handler)
    set_logger(logger)

    # Handle form submission
    if submit_button:
        if uploaded_file is not None:
            start_time = time.time()
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            with st.spinner("Processing..."):
                logger.info("Starting transcription process...")
                logger.info(get_gpu_info())
                logger.info(get_cpu_info())
                full_text, srt_content = transcribe_audio(model_name, tmp_file_path, language=language)

            if full_text and srt_content:
                # Output section
                
                with st.form(key='output_form'):
                    st.header("Output")        
                    # col1, col2 = st.columns(1)

                    # with col1:
                    st.subheader("Full Transcription")
                    st.text_area("", value=full_text, height=200)

                    st.subheader("Detected Language")
                    st.write(language)

                    # with col2:
                    st.subheader("Subtitles (SRT format)")
                    st.text_area("", value=srt_content, height=200)

                    logger.info("Processing completed successfully") 

                    submitted = st.form_submit_button("Download Subtitles")
                    if submitted:
                        st.download(
                            data=srt_content,
                            file_name="subtitles.srt",
                            mime="text/plain"
                        )
                    # # Add download button for subtitles
                    # st.download_button(
                    #     label="Download Subtitles",
                    #     data=srt_content,
                    #     file_name="subtitles.srt",
                    #     mime="text/plain"
                    # )
                
                end_time = time.time()
                total_time = end_time - start_time
                logger.info(f"Total processing time: {total_time:.2f} seconds")

            else:
                logger.error("Transcription failed. No text or subtitle was generated.")
                st.error("Transcription failed. No text or subtitle was generated.")

            os.unlink(tmp_file_path)
        else:
            logger.warning("No file uploaded")
            st.error("Please upload an audio/video file")

if __name__ == "__main__":
    main()
