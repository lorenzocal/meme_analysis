from get_meaning import configure_meaning_model, generate_meaning
import csv
import os


"""
This script reads from the image_captions.csv file and generates meanings for each image.
"""

def main():
    print("Configuring meaning generation model...")
    configure_meaning_model()
    print("Model configured successfully.")

    # Input and output file paths
    captions_path = "image_captions.csv"
    results_path = "meme_meanings.csv"
    
    if not os.path.exists(captions_path):
        print(f"Error: {captions_path} not found. Please run image captioning first.")
        return

    results = [["Meme", "Instance", "Caption", "Meaning"]]
    
    print("\nReading captions and generating meanings...")
    with open(captions_path, mode='r', encoding='utf-8') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row
        
        for row in reader:
            meme, instance, caption = row
            print(f"\nProcessing caption for {meme}/{instance}")
            print(f"Caption: {caption}")
            
            try:
                meaning = generate_meaning(caption)
                results.append([meme, instance, caption, meaning])
                print(f"Generated meaning: {meaning}\n")
                
            except Exception as e:
                print(f"Error generating meaning: {str(e)}")
                results.append([meme, instance, caption, f"Error: {str(e)}"])

    # Save results
    with open(results_path, mode="w", newline="", encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerows(results)
        print(f"\nResults saved to '{results_path}'")

if __name__ == "__main__":
    main() 