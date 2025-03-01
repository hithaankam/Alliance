import torch
import gc
import os
import streamlit as st
from torchvision import transforms
from transformers import SamModel, SamProcessor
from diffusers import AutoPipelineForInpainting
from PIL import Image

#  Set environment variable to avoid memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

#  Clear GPU memory before execution
torch.cuda.empty_cache()
gc.collect()

#  Check GPU memory availability and move to CPU if needed
device = "cuda" if torch.cuda.is_available() and torch.cuda.memory_reserved() < 10 * 1024**3 else "cpu"

#  Load SAM model and processor (force CPU if GPU is full)
model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50").to(device).to(torch.float16 if device == "cuda" else torch.float32)
processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")

# Streamlit UI Setup
st.title("FixIt: AI-Powered Clothing Inpainting")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    img = img.resize((512, 512))  # Resize image to optimize memory
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Define input points
    input_points = [[[256, 256]]]  # Adjust coordinates based on new image size
    inputs = processor(img, input_points=input_points, return_tensors="pt").to(device).to(torch.float16 if device == "cuda" else torch.float32)

    # Run the model (Move back to CPU if GPU is full)
    with torch.no_grad():
        if device == "cuda":
            try:
                outputs = model(**inputs)
            except torch.cuda.OutOfMemoryError:
                st.warning("🚨 GPU ran out of memory! Moving model to CPU...")
                model.to("cpu")
                device = "cpu"
                inputs = inputs.to("cpu")
                outputs = model(**inputs)
        else:
            outputs = model(**inputs)

    # Extract mask tensors
    masks = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
    )

    # Convert mask to PIL image
    to_pil = transforms.ToPILImage()
    mask_1 = to_pil(masks[0][0][2].to(dtype=torch.uint8) * 255)
    st.image(mask_1, caption="Generated Mask", use_column_width=True)

    # Unload SAM model from GPU (if needed)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Load inpainting pipeline with memory optimization
    pipeline = AutoPipelineForInpainting.from_pretrained(
        "redstonehero/ReV_Animated_Inpainting",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipeline.enable_model_cpu_offload()

    # Inpainting example with a prompt
    prompt = st.text_input("Enter a clothing design prompt", "flower-print, t-shirt")

    if st.button("Generate Inpainted Image"):
        with st.spinner("Generating image..."):
            image1 = pipeline(
                prompt=prompt,
                width=512,
                height=512,
                num_inference_steps=28,
                image=img,
                mask_image=mask_1,
                guidance_scale=3,
                strength=1.0
            ).images[0]
        st.image(image1, caption="Inpainted Image", use_column_width=True)

# Free GPU memory after execution
torch.cuda.empty_cache()
gc.collect()