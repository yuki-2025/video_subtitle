import torch
import datetime
import numpy as np
from transformers import WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer
from pydub import AudioSegment
import logging

# Set the path to FFmpeg and FFprobe
ffmpeg_path = "/usr/bin/ffmpeg"
ffprobe_path = "/usr/bin/ffprobe"

# Set the paths for pydub
AudioSegment.converter = ffmpeg_path
AudioSegment.ffmpeg = ffmpeg_path
AudioSegment.ffprobe = ffprobe_path

# Initialize logger
logger = logging.getLogger(__name__)

def set_logger(new_logger):
    global logger
    logger = new_logger

def format_timestamp(milliseconds):
    """Convert milliseconds to SRT timestamp format."""
    delta = datetime.timedelta(milliseconds=milliseconds)
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds % 1000:03d}"

def generate_srt(transcriptions, chunk_length_ms, subtitle_duration_ms=5000):
    """Generate SRT content from transcribed chunks with specified subtitle duration."""
    srt_output = ""
    srt_index = 1
    for i, chunk_text in enumerate(transcriptions):
        chunk_start_time = i * chunk_length_ms
        chunk_end_time = (i + 1) * chunk_length_ms
        
        # Split chunk text into words
        words = chunk_text.split()
        
        # Calculate number of subtitles for this chunk
        num_subtitles = max(1, int(chunk_length_ms / subtitle_duration_ms))
        words_per_subtitle = max(1, len(words) // num_subtitles)
        
        for j in range(0, len(words), words_per_subtitle):
            subtitle_words = words[j:j+words_per_subtitle]
            subtitle_text = " ".join(subtitle_words)
            
            start_time = chunk_start_time + (j // words_per_subtitle) * subtitle_duration_ms
            end_time = min(start_time + subtitle_duration_ms, chunk_end_time)
            
            srt_output += f"{srt_index}\n"
            srt_output += f"{format_timestamp(start_time)} --> {format_timestamp(end_time)}\n"
            srt_output += f"{subtitle_text}\n\n"
            
            srt_index += 1

    return srt_output

def transcribe_audio(model_name, audio_path, language='en', chunk_length_ms=30000):
    try:
        # Check if CUDA is available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")

        # Load model and processor
        model = WhisperForConditionalGeneration.from_pretrained(model_name).to(device)
        processor = WhisperProcessor.from_pretrained(model_name)
        tokenizer = WhisperTokenizer.from_pretrained(model_name)

        # Load audio
        audio = AudioSegment.from_file(audio_path)

        # Resample to 16000 Hz
        audio = audio.set_frame_rate(16000)

        # Initialize lists to store chunk transcriptions
        chunk_transcriptions = []

        # Process audio in chunks
        for i in range(0, len(audio), chunk_length_ms):
            chunk = audio[i:i+chunk_length_ms]

            # Convert chunk to numpy array
            chunk_array = np.array(chunk.get_array_of_samples()).astype(np.float32)

            # Normalize
            chunk_array = chunk_array / np.max(np.abs(chunk_array))

            # Process audio chunk
            input_features = processor(chunk_array, sampling_rate=16000, return_tensors="pt").input_features
            input_features = input_features.to(device)

            # Generate token ids
            forced_decoder_ids = tokenizer.get_decoder_prompt_ids(language=language, task="transcribe")
            predicted_ids = model.generate(input_features, forced_decoder_ids=forced_decoder_ids)

            # Decode token ids to text
            transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
            chunk_text = transcription[0].strip()
            chunk_transcriptions.append(chunk_text)

            # Print chunk transcription in real-time
            print(f"Chunk {i // chunk_length_ms + 1} transcription:")
            print(chunk_text)
            print("-" * 50)

        # Combine all chunk transcriptions
        full_text = " ".join(chunk_transcriptions)

        # Generate SRT content with 5-second subtitles
        srt_content = generate_srt(chunk_transcriptions, chunk_length_ms, subtitle_duration_ms=5000)

        return full_text, srt_content

    except Exception as e:
        logger.error(f"An error occurred during transcription: {str(e)}")
        return None, None
 