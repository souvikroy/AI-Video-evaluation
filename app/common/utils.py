import os
import time
import cv2
import parselmouth

from openai import OpenAI
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


def confidence_retrival(audio_path, logger):
    start = time.time()
    sound = parselmouth.Sound(audio_path)

    # Extract prosodic features
    pitch = sound.to_pitch()
    intensity = sound.to_intensity()
    harmonicity = sound.to_harmonicity()

    # Calculate confidence score based on prosodic features
    confidence_score = 0
    max_p = max_i = max_h = 0

    for i in range(pitch.get_number_of_frames()):
        pitch_value = pitch.get_value_at_time(int(i * pitch.get_time_step()))
        intensity_value = intensity.get_value(int(i * intensity.get_time_step()))
        harmonicity_value = harmonicity.get_value(int(i * harmonicity.get_time_step()))
        max_p = pitch_value if pitch_value > max_p else max_p
        max_i = intensity_value if intensity_value > max_i else max_i
        max_h = harmonicity_value if harmonicity_value > max_h else max_h
        if pitch_value > 0 and intensity_value > 0 and harmonicity_value > 0:
            confidence_score += pitch_value * intensity_value * harmonicity_value

    # Normalize the confidence score
    d = pitch.get_number_of_frames() * max_p * max_i * max_h
    if d:
        confidence_score /= d
    else:
        confidence_score = 0
    logger.info("Calculated Confidence Score in " + str(time.time()-start))
    return {'confidence': round(confidence_score * 100)}


def content_generation(topic: str, logger):
    logger.info("Generating Content for comparison")
    start = time.time()
    client = OpenAI(api_key=os.environ.get("OpenAI_API_KEY"))
    response = client.chat.completions.create(model="gpt-3.5-turbo", messages=[
        {'role': 'user',
         'content': f"Write a short description about {topic}."
         }
    ], temperature=1, max_tokens=256, top_p=1, frequency_penalty=0, presence_penalty=0)
    end = time.time()
    logger.info("Content Generated for comparison in " + str(round(end-start, 2)) + "s")
    return response.choices[0].message.content


def calculate_similarity(lecture_txt: str, reference_txt: str, logger):
    logger.info("Started content comparison")
    start = time.time()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    lecture_txt_emb = model.encode(lecture_txt)
    reference_txt_emb = model.encode(reference_txt)
    end = time.time()
    logger.info("content comparison finished in " + str(round(end-start, 2)) + "s")
    return cosine_similarity([lecture_txt_emb, reference_txt_emb])[1, 0]


def video_capture(emotion_function, video_path: str, logger, frequency: int = 100):
    """_summary_

    Args:
        emotion_function (function)
        frequency (int)

    Returns:
        emotion_lst: list
    """
    logger.info("Preparing emotion_dictionary...")
    start = time.time()
    vid = cv2.VideoCapture(video_path)
    emotion_lst = []
    length = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    jump = length // frequency
    for selected_frame in range(jump, length - jump - jump, jump):
        vid.set(cv2.CAP_PROP_POS_FRAMES, selected_frame - 1)
        res, frame = vid.read()
        emotion_lst.append(emotion_function(frame))

    vid.release()
    cv2.destroyAllWindows()
    counter = Counter(emotion_lst)
    end = time.time()
    logger.info("Prepared emotion_dictionary in " + str(round(end-start, 2)) + "s")
    return dict(counter)


def grammer_score_func(txt, logger):
    logger.info("Started grammer_score_func")
    start = time.time()
    client = OpenAI(api_key=os.environ.get("OpenAI_API_KEY"))
    response = client.chat.completions.create(model="gpt-3.5-turbo",
                                              messages=[
                                                  {'role': 'system',
                                                   'content': """You are a english teacher who score a text from 0 to 5 based on the following criteria.
     **0 - Poor**: Text contains numerous grammatical errors, making it difficult to comprehend. Errors may include incorrect word usage, subject-verb agreement issues, or inconsistent tense usage.
     **1 - Below Average**: Text has several grammatical errors that hinder understanding. Errors are noticeable and distract from the overall message. Basic grammatical structures may be incorrectly used.
     **2 - Average**: Text generally follows grammatical rules but contains occasional errors. Some sentences may lack clarity due to minor grammatical mistakes. Overall, the text is understandable but could be improved.
     **3 - Good**: Text demonstrates solid grasp of grammar with few errors. Sentences are clear and effectively convey the intended message. Minor errors may be present but do not significantly detract from comprehension.
     **4 - Very Good**: Text exhibits strong command of grammar with rare errors. Sentences are well-structured and articulate. The text flows smoothly and effectively communicates ideas.
     **5 - Excellent**: Text is virtually error-free in terms of grammar. Sentences are precise, concise, and grammatically correct. The writing demonstrates mastery of language conventions and enhances readability.

     YOU WILL ONLY RETURN A INTEGER, WHICH IS THE SCORE. NOTHING ELSE"""},
                                                  {
                                                      'role': 'user',
                                                      'content': f"Score the following text: {txt}."
                                                  }
                                              ],
                                              temperature=0, max_tokens=10, top_p=1, frequency_penalty=0,
                                              presence_penalty=0)
    response = str(response.choices[0].message.content)

    for i in response:
        if i.isnumeric():
            end = time.time()
            logger.info("grammer_score calculated in " + str(round(end-start, 2)) + "s")
            return int(i)

    end = time.time()
    logger.info("grammer_score calculated in " + str(round(end-start, 2)) + "s")
    return None


def emotion_score(emotion_dictionary: dict, logger):
    """_summary_

    Args:
        emotion_dictionary (dict): _description_

    Returns:
        _type_: _description_
    """
    lst = []
    for key, value in emotion_dictionary.items():
        lst.append((key, value))
    lst = sorted(lst, key=lambda x: x[-1], reverse=True)
    lst = [i[0] for i in lst][:2]
    logger.info("Calculated emotion_score")
    if ('Anger' in lst) or ('Sad' in lst):
        return 5
    else:
        return 10
