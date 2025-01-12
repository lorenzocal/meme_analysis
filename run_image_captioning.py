from PIL import Image
from image_captioning import load_image_captioning_model, generate_caption
import os
import csv

def main():
    print("Loading image captioning model...")
    load_image_captioning_model()
    print("Model loaded successfully.")

    dataset_path = "Dataset"
    results_path = "image_captions.csv"
    results = [["Meme", "Instance", "Caption"]]

    for meme in os.listdir(dataset_path):
        meme_path = os.path.join(dataset_path, meme)
        
        if os.path.isdir(meme_path):
            print(f"\nProcessing meme category: {meme}")

            for image in os.listdir(meme_path):
                image_path = os.path.join(meme_path, image)
                
                if image.lower().endswith((".jpg", ".jpeg", ".png")):
                    print(f"Processing image: {image}")
                    
                    try:
                        img = Image.open(image_path).convert("RGB")
                        caption = generate_caption(img)
                        
                        data_row = [meme, image, caption]
                        results.append(data_row)
                        
                        print(f"Caption generated: {caption}\n")
                        
                    except Exception as e:
                        print(f"Error processing {image}: {str(e)}")

    with open(results_path, mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(results)
        print(f"\nResults saved to '{results_path}'")

if __name__ == "__main__":
    main() 