import logging
import traceback
import concurrent.futures

from dotenv import load_dotenv

from app.common.utils import *
from app.services.audio_transcribe import *
from app.services.data_processing import Data_processing
from app.services.emotion_detection import EmotionDetection

load_dotenv()

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)


def __audio_task_handler(video_path, audio_path, demo_content):
    audio_extraction_status = Data_processing(video_path=video_path, audio_path=audio_path)

    if audio_extraction_status:
        logger.info("Audio extraction successful")
        txt = transcribe(audio_path=audio_path)
        if txt == "":
            raise Exception('No utterances found!!')
        if demo_content is not None:
            response = content_generation(topic=demo_content, logger=logger)
            # Calculating similarity
            similarity_score = calculate_similarity(lecture_txt=txt, reference_txt=response, logger=logger)
        else:
            similarity_score = 0
        # Calculate grammar score
        grammar_score = grammer_score_func(txt=txt, logger=logger)
        return {
            'similarity_score': round(similarity_score * 10),
            'grammar_score': round(grammar_score * 2)}
    else:
        raise Exception('Audio extraction Failed!!')


def __cv_task_handler(video_path):
    emotion_dictionary = video_capture(emotion_function=EmotionDetection, video_path=video_path, logger=logger)
    emotion_score_value = emotion_score(emotion_dictionary=emotion_dictionary, logger=logger)
    return {'emotion_score': emotion_score_value}


def __format_output(future_lst):
    response_lst = {
        'confidence': future_lst['confidence'] if future_lst['confidence'] > 0 else 0,
        'similarity_score': future_lst['similarity_score'] if future_lst['similarity_score'] > 0 else 0,
        'emotion_score': future_lst['emotion_score'] if future_lst['emotion_score'] > 0 else 0,
        'grammar_score': future_lst['grammar_score'] if future_lst['grammar_score'] > 0 else 0
    }
    return response_lst


def main(video_path: str, audio_path: str, demo_content: str):
    """
    Args:
        video_path (str): Path of the video file
        audio_path (str): Path of the audio file to save in a specific directory
    """
    try:
        future_lst = {}
        futures = []

        executor = concurrent.futures.ThreadPoolExecutor(2)

        future_1 = executor.submit(__audio_task_handler, video_path, audio_path, demo_content)
        futures.append(future_1)
        future_2 = executor.submit(__cv_task_handler, video_path)
        futures.append(future_2)
        future_3 = executor.submit(confidence_retrival, audio_path, logger)
        futures.append(future_3)

        concurrent.futures.wait(futures)
        for future in futures:
            future_lst.update(future.result())

        response_lst = __format_output(future_lst)
        return response_lst

    except Exception:
        logger.error(traceback.format_exc())
        return {"message": {
                        'confidence': 0,
                        'similarity_score': 0,
                        'emotion_score': 0,
                        'grammar_score': 0}}
