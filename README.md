# Alliance
# CoutureAI: Clothing Image Generator Using Stable Diffusion Pipeline

## Team: FixIt
**Team Members:**
- D Sri Neha
- Navya Sree Balu
- Hitha Ankam

## Project Overview
CoutureAI is an AI-powered clothing visualization tool designed to enhance the online shopping experience by allowing users to visualize their custom clothing designs. The project leverages **Stable Diffusion**, **Streamlit UI**, and **FastAPI backend** to generate high-quality clothing images from text descriptions. This helps consumers, fashion designers, and retailers bridge the gap between imagination and reality in fashion design.

### Key Features
- **AI-Driven Clothing Visualization**: Uses Stable Diffusion to generate realistic clothing images.
- **Custom Design Generation**: Users can input detailed clothing descriptions.
- **Interactive Refinement**: Provides AI-powered suggestions for modifications.
- **User-Friendly UI**: Built using Streamlit for seamless interaction.
- **E-Commerce Integration**: Potential for collaboration with retailers and fashion designers.
- **Inpainting & Masking**: Enhances image generation with AI-powered refinement.

### Target Users
- **Online Shoppers** who seek custom clothing options.
- **Fashion Designers** looking to visualize their designs.
- **Retailers & E-Commerce Platforms** aiming for AI-powered product customization.
- **Tailors & Custom Apparel Businesses** enhancing customer collaboration.

---
## File Structure
```
CoutureAI/
│── stable_diffusion_inpainting.ipynb        # Colab notebook for masking and inpainting
│── main.ipynb           # Colab notebook for running the pretrained model
│── vs code/            # Codebase for local execution
│   ├── app.py          # Main application script for UI and API integration
│   ├── config.py       # Configuration settings
│   ├── inpainting.py   # AI-driven inpainting functions
│   ├── model.py        # Model definitions and inference logic
│   ├── requirements.txt # Dependencies and installation instructions
│   ├── utils.py        # Helper functions for image processing
│── README.md           # Project documentation
```

## Setup & Installation
### 📌 Step 1: Check CUDA Availability & Clear Cache
```python
import torch

print("CUDA Available:", torch.cuda.is_available())

# Clear CUDA cache to free up memory
torch.cuda.empty_cache()
```

### 📌 Step 2: Install Required Packages (Run Once)
```python
!pip install -q streamlit diffusers transformers accelerate torch nest_asyncio
!pip install -q nest_asyncio
```

### 📌 Step 3: Import Required Libraries
```python
import torch
import streamlit as st
```


## Usage
- **Google Colab (Gen_Ai.ipynb)**: Run this for AI-based masking and inpainting.
- **Google Colab (main.ipynb)**: Load the pretrained model for direct inference.
- **VS Code (app.py)**: Start the UI and API for local execution.

## Challenges & Optimizations
- **Image Generation Performance**: Optimized Stable Diffusion for faster results.
- **Masking & Inpainting Accuracy**: Improved AI-generated designs with refined models.

## License
This project is licensed under the MIT License.

---
Feel free to contribute or provide feedback on our implementation!
