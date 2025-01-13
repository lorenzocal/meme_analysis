import os
import sys
sys.path.append(os.path.abspath("./CFSP"))
from CFNCFSP import CFNParser
import json
import base64
from typing import Dict, Optional
from dotenv import load_dotenv
import openai
from openai import OpenAI
import time

# Load environment variables
load_dotenv()

# Configure OpenAI
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

def encode_image(image_path: str) -> str:
    """
    Encode image to base64 string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path: str, max_retries: int = 3) -> Optional[Dict]:
    """
    Analyze an image using GPT-4V to get caption, OCR, and meaning.
    Includes retry logic for API calls.
    """
    for attempt in range(max_retries):
        try:
            # Encode image
            base64_image = encode_image(image_path)
            
            # Prepare the messages for analysis
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "I need three types of analysis for this meme image all in simplified Chinese:\n"
                                    "1. A detailed caption describing the visual elements\n"
                                    "2. OCR extraction (provide both original and translation)\n"
                                    "3. An analysis of the meme's meaning, including cultural references and humor\n\n"
                                    "Please format your response as JSON with these keys: caption, ocr, meaning"
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
                model="gpt-4o",  # Using the vision-specific model
                messages=messages,
                max_tokens=1000,
                temperature=0.7,
                response_format={"type": "json_object"}  # Ensure JSON response
            )

            # Parse and return the response
            analysis = json.loads(response.choices[0].message.content)
            print(analysis)
            print(analysis['meaning'])
            # Split meaning into sentences and analyze each one
            parser = CFNParser()
            meaning_text = analysis['meaning']
            
            # Split meaning text into sentences
            sentences = meaning_text.split('。')  # Split on Chinese period
            sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty strings and whitespace
            
            all_frames = []
            for sentence in sentences:
                if sentence:  # Skip empty sentences
                    results = parser.pipeline([sentence])
                    if results and results[0].parsing:
                        # Extract frame information from each parsing result
                        for parsing in results[0].parsing:
                            if parsing.target and parsing.target.frame:
                                frame_info = {
                                    "sentence": sentence,
                                    "target": parsing.target.word,
                                    "frame": parsing.target.frame.frame_name,
                                    "arguments": []
                                }
                                if parsing.arguments:
                                    for arg in parsing.arguments:
                                        if arg.fe:
                                            frame_info["arguments"].append({
                                                "role": arg.fe.fe_name,
                                                "text": sentence[arg.start:arg.end + 1]
                                            })
                                all_frames.append(frame_info)
            
            print("Frames extracted from each sentence:")
            for frame in all_frames:
                print(frame)
                
            analysis['frames'] = all_frames

            return analysis
        


        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2  # Exponential backoff
                print(f"Attempt {attempt + 1} failed. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            print(f"Error analyzing image {image_path}: {str(e)}")
            return None
        
       

def main():
    # Test with a single image
    test_image_path = "Dataset/Take my money/acb (1).jpg"  # Using the same test image
    
    if not os.path.exists(test_image_path):
        print(f"Error: Image not found at {test_image_path}")
        return
    
    print(f"Analyzing image: {test_image_path}")
    analysis = analyze_image(test_image_path)
    
    if analysis:
        print("\nAnalysis Results:")
        print("\nCaption:")
        print(analysis.get('caption', 'No caption key found.'))
        print("\nOCR:")
        print(analysis.get('ocr', 'No ocr key found.'))
        print("\nMeaning:")
        print(analysis.get('meaning', 'No meaning key found.'))
        
        # Save results to a JSON file
        output_file = "test_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        print(f"\nResults saved to {output_file}")
    else:
        print("Analysis failed.")

if __name__ == "__main__":
    main()