# Meme Analysis

This project analyzes memes using image captioning, meaning generation, and frame semantic analysis.

## Setup Instructions

1. Create and activate a Python virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
   - Create a `.env` file in the project root
   - Add your Google API key:
   ```
   GOOGLE_API_KEY=your_api_key_here
   ```

4. Prepare your dataset:
   - Place your meme images in the `Dataset` directory
   - Organize images in subdirectories by meme category
   Example structure:
   ```
   Dataset/
   ├── Distracted boyfriend/
   │   ├── image1.jpg
   │   └── image2.jpg
   └── Take my money/
       ├── image1.jpg
       └── image2.jpg
   ```

## Running the Analysis

You can run the analysis in different ways:

1. Complete analysis (captions, meanings, and frames):
```bash
python main.py
```

2. Image captioning only:
```bash
python run_image_captioning.py
```

3. Meaning generation only from captions:
```bash
python run_meaning_generation.py
```

4. Combined caption and meaning analysis:
```bash
python run_meme_analysis.py
```

## Output Files

The scripts generate different CSV files:
- `results.csv`: Complete analysis with frames
- `image_captions.csv`: Image captions only
- `meme_meanings.csv`: Captions with meanings
- `meme_analysis.csv`: Combined caption and meaning analysis

## Notes

- Make sure the virtual environment is activated before running any scripts
- The image captioning model will download automatically on first run
- Ensure you have a valid Google API key for meaning generation
