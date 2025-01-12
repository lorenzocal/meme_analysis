import os
from dotenv import load_dotenv
import google.generativeai as genai

def configure_meaning_model():
    global large_language_model
    
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment variable
    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    genai.configure(api_key=api_key)
    large_language_model = genai.GenerativeModel("gemini-1.0-pro")

def generate_meaning(caption):
    response = large_language_model.generate_content(
        f"Explain me this meme: {caption}. 80 words maximum",
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,  # Lower temperature for more focused responses
            max_output_tokens=150,
            top_p=0.8,
            top_k=40,
        ),
    )
    return response.text