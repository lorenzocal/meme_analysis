from PIL import Image
from image_captioning import load_image_captioning_model, generate_caption
from get_meaning import configure_meaning_model, generate_meaning
from extract_frames import load_frame_extractor, get_frames
import os
import csv

models_loaded = False
loaded_models = None 

def load_models():

    global models_loaded, loaded_models

    if models_loaded:
        print("Models are already loaded. Skipping execution.")
    else:
        print("Loading models...")
    
        load_image_captioning_model()
        configure_meaning_model()
        load_frame_extractor()

        models_loaded = True 

        print("Models loaded successfully.")

def main():

    load_models()

    dataset_path = "Dataset"
    results_path = "results.csv"
    results = [["Meme", "Instance", "Frames"]]

    for meme in os.listdir(dataset_path):
        meme_path = os.path.join(dataset_path, meme)
    
    # Ensure it's a folder
        if os.path.isdir(meme_path):
            print(f"Processing meme: {meme}")

            # Loop through each instance of the meme
            for image in os.listdir(meme_path):
                image_path = os.path.join(meme_path, image)
            
                # Check if it's an image file (e.g., with .jpg, .png extension)
                if image.lower().endswith((".jpg", ".jpeg", ".png")):

                    print(f"Processing instance: {image}")
                
                    img = Image.open(image_path).convert("RGB")


                    caption = generate_caption(img)
                    meaning = generate_meaning(caption)
                    frame_set = get_frames(meaning)

                    frames_string = ",".join(frame_set)

                    data_row = [meme, image, frames_string]
                    results.append(data_row)
                

    with open(results_path, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerows(results)
        print(f"CSV file '{results_path}' created and populated successfully.")

main()