import sys
import os
sys.path.append(os.path.abspath("./CFSP"))
from CFNCFSP import CFNParser
import json
import base64
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
from dotenv import load_dotenv
import openai
from openai import OpenAI
import time

# Load environment variables
load_dotenv()

# Configure OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# Initialize CFSP parser
cfn_parser = CFNParser()

def encode_image(image_path: str) -> str:
    """
    Encode image to base64 string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path: str, max_retries: int = 3) -> Optional[Dict]:
    """
    Analyze an image using GPT-4o to get caption, OCR, and meaning.
    Includes retry logic for API calls.
    """
    for attempt in range(max_retries):
        try:
            # Encode image
            base64_image = encode_image(image_path)
            
            # Prepare the messages for each analysis
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "I need three types of analysis for this meme image:\n1. A detailed caption describing the visual elements\n2. OCR extraction (if text is not in English, provide both original and translation)\n3. An analysis of the meme's meaning, including cultural references and humor\n\nPlease format your response as JSON with these keys: caption, ocr, meaning"
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

            # Make API call
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                response_format={"type": "json_object"}
            )

            # Parse the response
            analysis = json.loads(response.choices[0].message.content)
            return analysis

        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            print(f"Error analyzing image {image_path}: {str(e)}")
            return None

def extract_frames(text: str) -> List[Dict]:
    """
    Extract frames from text using CFSP parser.
    """
    try:
        # Use CFSP to extract frames
        frames = cfn_parser.parse(text)
        return frames
    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
        return []

def process_meme_directory(base_dir: str, output_dir: str = "results"):
    """
    Process all memes in the directory structure and save results to CSV and individual JSON files.
    """
    # Validate input directory
    if not os.path.exists(base_dir):
        print(f"Error: Directory {base_dir} does not exist")
        return

    results = []
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    subdirs = {
        'captions': os.path.join(output_dir, "captions"),
        'ocr': os.path.join(output_dir, "ocr"),
        'meanings': os.path.join(output_dir, "meanings")
    }
    
    for directory in subdirs.values():
        os.makedirs(directory, exist_ok=True)
    
    # Process images
    for root, dirs, files in os.walk(base_dir):
        category = os.path.basename(root)
        
        image_files = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        total_files = len(image_files)
        
        for idx, file in enumerate(image_files, 1):
            image_path = os.path.join(root, file)
            print(f"Processing {category}/{file} ({idx}/{total_files})")
            
            # Analyze image
            analysis = analyze_image(image_path)
            if analysis:
                try:
                    # Extract frames from meaning
                    frames = extract_frames(analysis['meaning'])
                    
                    # Create base filename for this image
                    base_filename = f"{category}_{os.path.splitext(file)[0]}"
                    
                    # Save individual JSON files
                    for key, subdir in zip(['caption', 'ocr', 'meaning'], subdirs.values()):
                        output_path = os.path.join(subdir, f"{base_filename}_{key}.json")
                        data = {key: analysis[key]}
                        if key == 'meaning':
                            data['frames'] = frames
                        
                        with open(output_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, ensure_ascii=False, indent=2)
                    
                    # Add to results for CSV
                    results.append({
                        'category': category,
                        'image': file,
                        'caption': analysis['caption'],
                        'ocr': analysis['ocr'],
                        'meaning': analysis['meaning'],
                        'frames': json.dumps(frames, ensure_ascii=False)
                    })
                
                except Exception as e:
                    print(f"Error processing results for {file}: {str(e)}")
                    continue
    
    # Save to CSV
    if results:
        try:
            df = pd.DataFrame(results)
            csv_path = os.path.join(output_dir, "meme_frames_zh.csv")
            df.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"Results saved to {csv_path}")
            print(f"Individual results saved in {output_dir} directory")
        except Exception as e:
            print(f"Error saving CSV file: {str(e)}")
    else:
        print("No results were generated")

if __name__ == "__main__":
    # Process memes from the Dataset directory
    process_meme_directory("Dataset") 