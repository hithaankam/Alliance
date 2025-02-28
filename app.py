
# app.py
import streamlit as st
import io
from config import MODEL_ID, DEFAULT_STEPS, MAX_STEPS
from utils import validate_token
from model import ImageGenerator

def init_session_state():
    """Initialize session state variables."""
    if 'hf_token' not in st.session_state:
        st.session_state.hf_token = None
    if 'generator' not in st.session_state:
        st.session_state.generator = None

def setup_page():
    """Configure page settings."""
    st.set_page_config(page_title="AI Image Generator", layout="wide")
    st.title("🎨 AI Image Generator")
    st.write("Generate images from text descriptions using Stable Diffusion")

def token_input():
    """Handle token input and validation."""
    with st.form("token_form"):
        token = st.text_input("Enter your Hugging Face token:", type="password")
        token_submit = st.form_submit_button("Submit Token")
        
        if token_submit:
            if validate_token(token):
                st.session_state.hf_token = token
                st.session_state.generator = ImageGenerator(token, MODEL_ID)
                st.success("Token validated successfully!")
                st.experimental_rerun()
            else:
                st.error("Invalid token. Please check your token and try again.")
    
    st.info("To get your Hugging Face token:\n1. Go to https://huggingface.co/settings/tokens\n2. Create a new token (READ access is sufficient)")
    st.stop()

def main():
    init_session_state()
    setup_page()
    
    # Check for token
    if st.session_state.hf_token is None:
        token_input()
        
    # Logout button
    if st.sidebar.button("Logout (Clear Token)"):
        st.session_state.hf_token = None
        st.session_state.generator = None
        st.experimental_rerun()
    
    # Generation form
    with st.form("generation_form"):
        prompt = st.text_area(
            "Enter your prompt:",
            placeholder="A serene landscape with mountains and a lake at sunset..."
        )
        
        num_steps = st.slider(
            "Number of inference steps:",
            min_value=10,
            max_value=MAX_STEPS,
            value=DEFAULT_STEPS,
            help="Higher values = better quality but slower generation"
        )
        
        submit_button = st.form_submit_button("Generate Image")
    
    # Handle image generation
    if submit_button and prompt:
        try:
            with st.spinner("Generating your image... This might take several minutes on CPU..."):
                generated_image = st.session_state.generator.generate(prompt, num_steps)
            
            # Display image
            st.image(generated_image, caption=f"Generated image for: {prompt}", 
                    use_column_width=True)
            
            # Download button
            buf = io.BytesIO()
            generated_image.save(buf, format="PNG")
            st.download_button(
                label="Download Image",
                data=buf.getvalue(),
                file_name="generated_image.png",
                mime="image/png"
            )
            
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            if "401" in str(e):
                st.error("Authentication error. Please check your Hugging Face token.")
                st.session_state.hf_token = None
                st.experimental_rerun()

    # Usage instructions
    with st.expander("How to use"):
        st.markdown("""
        1. Enter a descriptive prompt in the text area
        2. Adjust the number of inference steps if desired (higher = better quality but slower)
        3. Click 'Generate Image' button
        4. Wait for the image to be generated (this may take several minutes on CPU)
        5. Download the generated image if desired
        
        Note: This application runs on CPU which means generation will be slower than on GPU.
        Consider reducing the number of inference steps for faster generation.
        """)

if __name__ == "__main__":
    main()