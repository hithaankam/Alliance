import torch
import requests
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from torchvision import transforms
from transformers import SamModel, SamProcessor
from diffusers import AutoPipelineForInpainting

# Function to load images from URLs
def load_image_from_url(url, size=(1024, 1024)):
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(BytesIO(response.content)).convert("RGB")
        return img.resize(size)
    else:
        raise ValueError(f"Failed to fetch image from {url}")

# Load images (replace with a local image if needed)
image_path = "C:/Users/Hanumanth/Downloads/white_tshirt.jpg"  # Update this path
mask_path = "C:/Users/Hanumanth/Downloads/white_tshirt.jpg"  # Update this path

image = Image.open(image_path).convert("RGB").resize((1024, 1024))
mask_image = Image.open(mask_path).convert("RGB").resize((1024, 1024))

# Display images
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.axis("off")
plt.title("Original Image")

plt.subplot(1, 2, 2)
plt.imshow(mask_image)
plt.axis("off")
plt.title("Mask Image")
plt.show()

# Load SlimSAM model for segmentation
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50").to(device)
processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")

# Define input points for segmentation
input_points = [[[320, 600]]]  # Adjust as needed

# Process input
inputs = processor(image, input_points=input_points, return_tensors="pt").to(device)
outputs = model(**inputs)

# Post-process masks
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)

# Convert mask to PIL
to_pil = transforms.ToPILImage()
binary_matrix = masks[0][0][2].to(dtype=torch.uint8) * 255
mask_1 = to_pil(binary_matrix)

# Display the generated mask
plt.imshow(mask_1)
plt.axis("off")
plt.title("Generated Mask")
plt.show()

# Load inpainting model
pipe = AutoPipelineForInpainting.from_pretrained(
    "stabilityai/stable-diffusion-2-inpainting",  # Alternative working model
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
).to(device)

# Define the prompt
prompt = "flower-print, t-shirt"

# Run inpainting pipeline
generator = torch.Generator(device=device).manual_seed(0)
output_image = pipe(
    prompt=prompt,
    image=image.resize((512, 768)),
    mask_image=mask_1.resize((512, 768)),
    guidance_scale=8.0,
    num_inference_steps=20,
    strength=0.99,
    generator=generator,
).images[0]

# Display and save the result
plt.imshow(output_image)
plt.axis("off")
plt.title("Inpainted Image")
plt.show()
output_image.save("output.png")

print("Inpainted image saved as 'output.png'")
