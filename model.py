import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from transformers import SamModel, SamProcessor
from diffusers import AutoPipelineForInpainting
from PIL import Image
import os

# Load the SlimSAM model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("Zigeng/SlimSAM-uniform-50").to(device)
processor = SamProcessor.from_pretrained("Zigeng/SlimSAM-uniform-50")

# Load a local image
image_path = r"C:/Users/Hanumanth/Downloads/white_tshirt.jpg"  # Replace with your actual image path
img = Image.open(image_path).convert("RGB")

# Define input points for segmentation
input_points = [[[320, 800]]]  # Adjust as needed

# Process image for segmentation
inputs = processor(img, input_points=input_points, return_tensors="pt").to(device)
outputs = model(**inputs)

# Extract mask tensors
masks = processor.image_processor.post_process_masks(
    outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)

# Convert mask tensor to PIL Image
to_pil = transforms.ToPILImage()
binary_matrix_1 = masks[0][0][2].to(dtype=torch.uint8) * 255
mask_1 = to_pil(binary_matrix_1)

# Save mask image
mask_1.save("mask_1.png")
print("Mask saved as 'mask_1.png'")

# Display the mask
plt.imshow(mask_1, cmap="gray")
plt.axis("off")
plt.title("Generated Mask")
plt.savefig("output.png", dpi=300, bbox_inches="tight")  # Save as high-quality image
plt.show()

# Repeat for second mask
input_points_2 = [[[200, 850]]]
inputs_2 = processor(img, input_points=input_points_2, return_tensors="pt").to(device)
outputs_2 = model(**inputs_2)
masks_2 = processor.image_processor.post_process_masks(
    outputs_2.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
)

# Convert second mask
binary_matrix_2 = masks_2[0][0][1].to(dtype=torch.uint8) * 255
mask_2 = to_pil(binary_matrix_2)
mask_2.save("mask_2.png")
print("Mask saved as 'mask_2.png'")

# Function to display images side by side
def make_image_grid(images, cols=3, rows=1):
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.axis("off")
    plt.show()

# Display images
make_image_grid([img, mask_1, mask_2], cols=3, rows=1)

# Load Stable Diffusion Inpainting model
pipeline = AutoPipelineForInpainting.from_pretrained(
    "runwayml/stable-diffusion-inpainting",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    variant="fp16" if device == "cuda" else None
).to(device)

# Load the saved mask for inpainting
init_image = img.resize((512, 512))
mask_image = Image.open("mask_1.png").convert("RGB").resize((512, 512))

# Run inpainting
generator = torch.Generator(device).manual_seed(92)
prompt = "blue t-shirt"
output_image = pipeline(prompt=prompt, image=init_image, mask_image=mask_image, generator=generator).images[0]

# Save and display the final image
output_image.save("inpainted_image.png")
print("Inpainted image saved as 'inpainted_image.png'")

make_image_grid([init_image, mask_image, output_image], rows=1, cols=3)

# Clear GPU memory (optional)
torch.cuda.empty_cache()
