import os
from pydub.silence import split_on_silence
from pydub import AudioSegment #Using AudioSegment object for removing silence
import soundfile # Play the audio file
from pydub.playback import play
import sys
import shutil # Removing the folder

def make_dir():
    # Create the folder to store the result
    try:
        os.mkdir(human_voice_16)
    except OSError:
        print ("Creation of the directory {} failed".format(human_voice_16))
    else:
        print ("Successfully created the directory {} ".format(human_voice_16))

    # Create the folder for segmentation
    try:
        os.mkdir(segment_folder)
    except OSError:
        print ("Creation of the directory {} failed".format(segment_folder))
    else:
        print ("Successfully created the directory {} ".format(segment_folder))

def convert_32_to_16():
    # convert the 32bit wav file to 16bit
    for file in os.listdir(human_voice_folder):
        data,sample_rate = soundfile.read(os.path.join(human_voice_folder,file))
        soundfile.write(os.path.join(human_voice_16,file),data,sample_rate,subtype='PCM_16')

def remove_silence(length):
    second = length*1000
    i = 0
    for file in os.listdir(human_voice_16):
        sound_obj = AudioSegment.from_wav(os.path.join(human_voice_16,file))
        # Change the sample rate from 44100 to 16000
        sound_obj = sound_obj.set_frame_rate(16000)
        # split_on_silence(audio_segment, min_silence_len=1000, silence_thresh=-16, keep_silence=100,seek_step=1)
        chunks = split_on_silence(sound_obj)
        # Create a empty audioSegment object to store non-silence segment
        combined = AudioSegment.empty()
        for chunk in chunks:
            combined+=chunk
        duration = combined.duration_seconds
        start = 0
        while start+length < duration:
            segment = combined[start*1000:start*1000+second]
            start = start+length
            filename = str(i)+'.wav'
            segment.export(os.path.join(segment_folder,filename),format="wav")
            i = i+1

def remove_folder(path):
    shutil.rmtree(path)

if __name__ == "__main__":

    # Input the path
    if len(sys.argv) < 2:
        print('Please enter the path of the music folder!')
        sys.exit()

    argv_1 = sys.argv[1]
    human_voice_folder = argv_1
    human_voice_16 = './Result_16'
    segment_folder = './Segment'
    # Prepare folder
    make_dir()
    # Convert 32bit wav to 16bit
    convert_32_to_16()
    # Remove silence and restore
    print('Finish converting 32 bit wav file to 16bit!')
    remove_silence(4)
    print('Finish removing silence part and segemnt!')
    remove_folder(human_voice_16)
    print('Finish removing the folder!')
