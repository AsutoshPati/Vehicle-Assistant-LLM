import audioop
import gc
import os
import time
import uuid
import wave

import numpy as np
import pyaudio
import sounddevice as sd
import torch
import whisperx
from pydub import AudioSegment

# from transformers import AutoModelForCausalLM, AutoTokenizer
from TTS.api import TTS

# Whisper Model Parameters
WHISPER_MODEL_NAME = "small.en"
COMPUTE_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# use "float16" for GPU & "int8" for CPU
COMPUTE_DTYPE = "float16" if COMPUTE_DEVICE == "cuda" else "int8"
SUPRESS_NUMERALS = True
BATCH_SIZE = 16

RECORD_DIR = "./recordings"

# Create a directory to store recordings if it doesn't exist
if not os.path.exists(RECORD_DIR):
    os.makedirs(RECORD_DIR)

# Load the whisper model
whisper_model = whisperx.load_model(
    WHISPER_MODEL_NAME,
    COMPUTE_DEVICE,
    compute_type=COMPUTE_DTYPE,
    # asr_options={"suppress_numerals": SUPRESS_NUMERALS}
)

# # Load TTS model and tokenizer
# model_name = "facebook/fastspeech2"
# model = AutoModelForCausalLM.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# tts_model = "tts_models/en/ljspeech/glow-tts"
# tts_model = "tts_models/en/ljspeech/tacotron2-DDC"
# tts_model = "tts_models/en/ljspeech/fast_pitch"
tts_model = "tts_models/en/ljspeech/speedy-speech"
tts = TTS(model_name=tts_model, progress_bar=True)


def normalize_audio(frames, target_rms=1000):
    # Adjust audio frames to a target RMS for improved sound quality
    rms = audioop.rms(b"".join(frames), 2)
    gain = target_rms / (rms + 1)  # Avoid division by zero
    normalized_frames = [audioop.mul(frame, 2, gain) for frame in frames]
    return normalized_frames


def record_audio(silence_threshold=1000, silence_duration=2):
    chunk = 1024
    audio_format = pyaudio.paInt16
    channels = 1
    rate = 44100

    p = pyaudio.PyAudio()
    stream = p.open(
        format=audio_format,
        channels=channels,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
    )

    print("Recording...")
    frames = []
    silence_count = 0

    while True:
        data = stream.read(chunk)
        frames.append(data)
        rms = audioop.rms(data, 2)
        if rms < silence_threshold:
            silence_count += 1
        else:
            silence_count = 0

        if silence_count >= silence_duration * (rate / chunk):
            print("Silence detected. Stopping recording.")
            break

    stream.stop_stream()
    stream.close()
    p.terminate()

    # Normalize audio
    normalized_frames = normalize_audio(frames)

    output_filename = str(uuid.uuid4()) + ".wav"
    output_filename = os.path.join(os.getcwd(), RECORD_DIR, output_filename)

    with wave.open(output_filename, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(p.get_sample_size(audio_format))
        wf.setframerate(rate)
        wf.writeframes(b"".join(normalized_frames))

    print(f"Audio saved to {output_filename}")
    return output_filename


def get_audio_len(audio_path):
    audio = AudioSegment.from_file(audio_path)
    length_seconds = 0.0
    if audio:
        length_ms = len(audio)
        length_seconds = length_ms / 1000.0

    print("Audio file size:", length_seconds)
    return length_seconds


def process_audio(audio_file):
    global BATCH_SIZE
    global whisper_model

    audio = whisperx.load_audio(audio_file)
    result = whisper_model.transcribe(audio, batch_size=BATCH_SIZE)
    # print(result["segments"])  # before alignment

    gc.collect()
    torch.cuda.empty_cache()
    # del whisper_model

    return result["segments"]


def get_text_from_transcribe(processed_transcribe):
    return " ".join([sentence["text"] for sentence in processed_transcribe])


def speech_to_text_process() -> str:
    print("Starting recording")
    recorded_file = record_audio(silence_threshold=500, silence_duration=2)
    print("Recording ended, processing the data")

    final_text = ""
    # Only process the text having at least 2 seconds of recording
    if get_audio_len(recorded_file) > 2:
        start_time = time.time()
        transcribe = process_audio(recorded_file)
        final_text = get_text_from_transcribe(transcribe)
        print(f"Transcribe took {time.time() - start_time} seconds")
        print("Transcribed text: ", final_text)

    # Delete the recording, after process is completed
    os.remove(recorded_file)

    return final_text


def text_to_speech_process(text: str):
    print("Generating speech")

    # only process text with more than 3 words
    if len(text.split(" ")) <= 3:
        print("Not enough word to generate speech")
        return

    start_time = time.time()
    audio_array = tts.tts(text)

    # Normalize audio data to the range [-1, 1]
    audio_array = audio_array / np.max(np.abs(audio_array))
    print(f"Audio generation took {time.time() - start_time} seconds")

    # Play the audio using sounddevice
    sd.play(audio_array, samplerate=22050)  # Adjust samplerate if necessary
    sd.wait()  # Wait until the audio is done playing

    gc.collect()


if __name__ == "__main__":
    try:
        while True:
            recorded_text = speech_to_text_process()
            text_to_speech_process(recorded_text)

            print("-------------- \n\n")
    except KeyboardInterrupt:
        print("Closing the program")
