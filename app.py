import streamlit as st
from src.hf_client import HuggingFaceClient
import io
import os

st.set_page_config(page_title="Music AI Generator", layout="wide")

st.title("🎵 Music AI Generator")
st.subheader("Generate music based on emotions using AI emotion recognition")

# Sidebar configuration
with st.sidebar:
    st.header("Configuration")
    # Try to use environment variable first
    default_api_key = os.getenv("HF_API_KEY", "")
    api_key = st.text_input("Hugging Face API Key", type="password", value=default_api_key, help="Enter your HF API key or set HF_API_KEY env variable")
    model_size = st.selectbox(
        "Music Model Size",
        ["medium", "small", "large"],
        help="small = faster, large = higher quality"
    )

st.divider()

# Create two tabs for different workflows
tab1, tab2 = st.tabs(["📤 Upload Audio for Emotion Detection", "🎭 Manual Emotion Selection"])

with tab1:
    st.subheader("Upload Audio to Detect Emotion")
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav", "mp3", "ogg", "flac"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format=f"audio/{uploaded_file.type}")
        
        if st.button("🔍 Detect Emotion", type="primary"):
            if not api_key:
                st.error("Please provide your Hugging Face API key")
            else:
                try:
                    client = HuggingFaceClient(api_key=api_key)
                    
                    with st.spinner("Analyzing emotion from audio..."):
                        audio_data = uploaded_file.read()
                        emotion_result = client.recognize_emotion(audio_data)
                    
                    st.success("Emotion detected!")
                    st.json(emotion_result)
                    
                    # Extract emotion label and generate music
                    if emotion_result:
                        st.subheader("Generating music based on detected emotion...")
                        detected_emotion = emotion_result[0].get("label", "calm") if isinstance(emotion_result, list) else "calm"
                        
                        prompt = client.get_emotion_prompt(detected_emotion)
                        st.info(f"Prompt: *{prompt}*")
                        
                        with st.spinner("Generating music..."):
                            music_data = client.generate_music(f"facebook/musicgen-{model_size}", prompt)
                        
                        st.success("Music generated!")
                        st.audio(music_data, format="audio/wav")
                        st.download_button(
                            label="Download Generated Music",
                            data=music_data,
                            file_name=f"music_{detected_emotion}.wav",
                            mime="audio/wav"
                        )
                except Exception as e:
                    st.error(f"Error: {str(e)}")

with tab2:
    st.subheader("Manual Emotion Selection")
    
    col1, col2 = st.columns(2)
    
    with col1:
        emotion = st.radio(
            "Choose an emotion:",
            ["Happy", "Sad", "Calm", "Energetic", "Romantic", "Dark", "Anger", "Fear", "Surprise"],
            horizontal=False
        )
    
    with col2:
        custom_prompt = st.text_area(
            "Or enter a custom music description:",
            placeholder="e.g., A smooth jazz song with piano and strings..."
        )
    
    if st.button("🎼 Generate Music", type="primary"):
        if not api_key:
            st.error("Please provide your Hugging Face API key")
        else:
            try:
                client = HuggingFaceClient(api_key=api_key)
                
                if custom_prompt:
                    prompt = custom_prompt
                else:
                    prompt = client.get_emotion_prompt(emotion)
                
                st.info(f"Generating music with prompt: *{prompt}*")
                
                with st.spinner("Generating music... This may take a moment"):
                    music_data = client.generate_music(f"facebook/musicgen-{model_size}", prompt)
                
                st.success("Music generated successfully!")
                st.audio(music_data, format="audio/wav")
                
                st.download_button(
                    label="Download Music",
                    data=music_data,
                    file_name=f"music_{emotion.lower()}.wav",
                    mime="audio/wav"
                )
            except Exception as e:
                st.error(f"Error generating music: {str(e)}")

# Footer
st.divider()
st.markdown("""
*Created with ❤️ using Streamlit and Hugging Face*

**Models Used:**
- Emotion Recognition: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
- Music Generation: `facebook/musicgen-{small|medium|large}`
""")
