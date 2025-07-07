from TTS.api import TTS
import sounddevice as sd
import gc
import os
import time
import numpy as np
from data_retrieval import query_rag

TTS_MODEL = os.environ.get("TTS_MODEL", "tts_models/en/ljspeech/speedy-speech")
tts = TTS(model_name=TTS_MODEL, progress_bar=True)


def text_to_speech_process(text: str):
    print("Generating speech...")

    # only process text with more than 3 words
    if len(text.split(" ")) <= 3:
        print("Not enough word to generate speech")
        return

    start_time = time.time()
    audio_array = tts.tts(text)

    # Normalize audio data to the range [-1, 1]
    audio_array = audio_array / np.max(np.abs(audio_array))
    print(f"Audio generation took {time.time() - start_time} seconds")

    gc.collect()
    return audio_array, tts.synthesizer.output_sample_rate


def get_ai_response(query):
    response = query_rag(query)

    # Return the audio response (audio array) & sample rate
    return text_to_speech_process(response["AI Response"])


if __name__ == "__main__":
    sample_text = """Thank you for exploring this code;
to start the assistant run main program"""

    audio_res, tts_sample_rate = text_to_speech_process(sample_text)

    # Play the audio using sounddevice
    sd.play(audio_res, samplerate=tts_sample_rate)
    sd.wait()   # Wait until the audio is done playing

