import os
import base64
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def analyze_image(image_path):
    # Convert image to base64
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode('utf-8')

    # Create message with the image
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What's in this image? Please describe it in detail."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                }
            ]
        }
    ]

    # Make the API call
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=300
    )

    # Print the response
    print("\nGPT's Analysis:")
    print(response.choices[0].message.content)

if __name__ == "__main__":
    # Test with an image
    image_path = "Dataset/Take my money/acb (1).jpg"  # Update this path to your image
    
    if os.path.exists(image_path):
        print(f"Analyzing image: {image_path}")
        analyze_image(image_path)
    else:
        print(f"Error: Image not found at {image_path}") 