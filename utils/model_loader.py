# Load environment variables and initialize Llama 2 locally
import os
import requests
from pathlib import Path
from langchain_community.llms import LlamaCpp
from dotenv import load_dotenv

load_dotenv()

def download_model(url, save_path):
    """Download the model file if it doesn't exist locally."""
    if not os.path.exists(save_path):
        print(f"Downloading model to {save_path}...")
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Stream the download to handle large files
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("Model download completed!")

def load_local_llm():
    model_url = "https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf"
    # Use absolute path based on the current file's location
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "models", "llama-2-7b.Q4_K_M.gguf")
    
    # Ensure model exists locally
    download_model(model_url, model_path)
    
    # Load quantized Llama 2 model
    return LlamaCpp(
        model_path=model_path,
        temperature=0.7,
        max_tokens=512,
        n_ctx=2048,
        verbose=True
    )
