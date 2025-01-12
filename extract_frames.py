from frame_semantic_transformer import FrameSemanticTransformer

def load_frame_extractor():
  
  global frame_transformer

  frame_transformer = FrameSemanticTransformer()

def get_frames(explaination):
  
    frame_set = set()
    sentences = explaination.split('.')

    for sentence in sentences:
        result = frame_transformer.detect_frames(sentence)
        #print(f'Results found in: "{result.sentence}"')
        for frame in result.frames:
            frame_set.add(frame.name)
    #print(f"\nFRAME: {frame.name}")
    #for element in frame.frame_elements:
        #print(f"\t{element.name}: {element.text}")

    return frame_set