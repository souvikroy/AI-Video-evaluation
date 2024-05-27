from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def transcribe(audio_path: str):
    """_summary_

    Args:
        audio_path (str): path of the audio file to be transcribed
    """
    model_id = "distil-whisper/distil-small.en"

    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    processor = AutoProcessor.from_pretrained(model_id)
    pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor
    )
    transcript = pipe(audio_path)
    return transcript['text']
