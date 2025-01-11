import google.generativeai as genai

def configure_meaning_model():

    global large_language_model

    genai.configure(api_key="AIzaSyBt4yTs7AzUgrkjrpUgCU5gkVz1_RKHckQ")
    large_language_model = genai.GenerativeModel("gemini-1.5-flash")

def generate_meaning(caption):
    response = large_language_model.generate_content(
    f"Explain me this meme: {caption}. 80 words maximum",
    generation_config=genai.types.GenerationConfig(
        #candidate_count=1,
        #stop_sequences=["x"],
        #max_output_tokens=100,
        temperature=1.5,
        ),
    )
    return response.text