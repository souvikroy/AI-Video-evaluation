from moviepy.editor import VideoFileClip


def Data_processing(video_path: str, audio_path: str):
    video_clip = VideoFileClip(video_path)
    audio_clip = video_clip.audio
    try:
        audio_clip.write_audiofile(audio_path)
        return True

    except Exception as e:
        print(e, "Error while saving...")
        return False
