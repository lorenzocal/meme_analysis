from PIL import Image
from image_captioning import load_image_captioning_model, generate_caption
from get_meaning import configure_meaning_model, generate_meaning
import os
import csv


"""
This scripts reads from images in the dataset and generates captions and meanings for each image.
"""
def main():
    print("Loading models...")
    load_image_captioning_model()
    configure_meaning_model()
    print("Models loaded successfully.")

    dataset_path = "Dataset"
    results_path = "meme_analysis.csv"
    results = [["Meme", "Instance", "Caption", "Meaning"]]

    for meme in os.listdir(dataset_path):
        meme_path = os.path.join(dataset_path, meme)
        
        if os.path.isdir(meme_path):
            print(f"\nProcessing meme category: {meme}")

            for image in os.listdir(meme_path):
                image_path = os.path.join(meme_path, image)
                
                if image.lower().endswith((".jpg", ".jpeg", ".png")):
                    print(f"Processing image: {image}")
                    
                    try:
                        # Generate caption
                        img = Image.open(image_path).convert("RGB")
                        caption = generate_caption(img)
                        print(f"Caption generated: {caption}")
                        
                        # Generate meaning
                        meaning = generate_meaning(caption)
                        print(f"Meaning generated: {meaning}\n")
                        
                        data_row = [meme, image, caption, meaning]
                        results.append(data_row)
                        
                    except Exception as e:
                        print(f"Error processing {image}: {str(e)}")

    with open(results_path, mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(results)
        print(f"\nResults saved to '{results_path}'")

if __name__ == "__main__":
    main() 