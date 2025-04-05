import os
import re

os.environ["OMP_NUM_THREADS"] = "4"

from data_retrieval import query_rag
from speech_recognition import speech_to_text_process, text_to_speech_process

known_name_mistakes = {"Altroz": ["Ultras", "ULTROACH"]}


def replace_deviations(text, data):
    # Create a reverse mapping for quick lookup
    deviation_map = {
        deviation: name
        for name, deviations in data.items()
        for deviation in deviations
    }

    # Function to replace deviations with actual names
    def replacer(match):
        word = match.group(0)
        # Replace if found, otherwise return original word
        return deviation_map.get(word, word)

    # Use regex to find words in the text and replace deviations
    pattern = r"\b(?:" + "|".join(map(re.escape, deviation_map.keys())) + r")\b"
    corrected_text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)

    return corrected_text


try:
    print("Starting main program")
    while True:
        # Get the transcribed text from speech
        voice_query = speech_to_text_process()

        # Replace the common known deviations
        cleaned_text = replace_deviations(voice_query, known_name_mistakes)

        # only process text with more than 3 words
        if len(cleaned_text.split(" ")) >= 3:
            # Get response using RAG
            response = query_rag(voice_query)

            # Speak out the response generated from RAG
            text_to_speech_process(response["AI Response"])

        print("\n\n-----------\n\n")

except KeyboardInterrupt:
    print("Closing main program")
    pass
