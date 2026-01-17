import os 
from pydub import AudioSegment
import pyaudio as pa
import os
from pydub import AudioSegment
import pyaudio

# Path to your audio file
file_path = "data\recordings\AUD-20250811-WA0012.mp3"

# Step 1: Check if file exists
if not os.path.exists(file_path):
    print(f"File not found: {file_path}")
    exit()

# Step 2: Load audio with pydub
audio = AudioSegment.from_file(file_path, format="mp3")

# Convert to raw audio data
raw_data = audio.raw_data
sample_width = audio.sample_width
frame_rate = audio.frame_rate
channels = audio.channels

# Step 3: Play audio with PyAudio
p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(sample_width),
                channels=channels,
                rate=frame_rate,
                output=True)

print("Playing audio...")
stream.write(raw_data)

stream.stop_stream()
stream.close()
p.terminate()

print("Playback finished.")
