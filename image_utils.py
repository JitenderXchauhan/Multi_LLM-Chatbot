import os
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()

def generate_image(prompt: str):
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise ValueError("HF_TOKEN not found. Please set it in .env")

    client = InferenceClient(
        provider="hf-inference",
        api_key=hf_token,
    )
    image = client.text_to_image(
        prompt,
        model="stabilityai/stable-diffusion-xl-base-1.0",
    )
    return image
