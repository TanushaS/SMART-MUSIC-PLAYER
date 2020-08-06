from cv2 import WINDOW_NORMAL
import random, glob
from pygame import mixer
import cv2
import time
from face_detect import find_faces
from image_commons import nparray_as_image, draw_with_alpha


def _load_emoticons(emotions):
    """
    Loads emotions images from graphics folder.
    :param emotions: Array of emotions names.
    :return: Array of emotions graphics.
    """
    l=[nparray_as_image(cv2.imread('graphics/%s.png' % emotion, -1), mode=None) for emotion in emotions]
    for emotion in emotions:
        print(emotion,nparray_as_image(cv2.imread('graphics/%s.png' % emotion, -1), mode=None))
    return l


def show_webcam_and_run(model, emoticons, window_size=None, window_name='webcam', update_time=10):
    """
    Shows webcam image, detects faces and its emotions in real time and draw emoticons over those faces.
    :param model: Learnt emotion detection model.
    :param emoticons: List of emotions images.
    :param window_size: Size of webcam image window.
    :param window_name: Name of webcam image window.
    :param update_time: Image update time interval.
    """
    cv2.namedWindow(window_name, WINDOW_NORMAL)
    if window_size:
        width, height = window_size
        cv2.resizeWindow(window_name, width, height)

    vc = cv2.VideoCapture(0)
    if vc.isOpened():
        read_value, webcam_image = vc.read()
    else:
        print("webcam not found")
        return

    while read_value:
        for normalized_face, (x, y, w, h) in find_faces(webcam_image):
            prediction = model.predict(cv2.resize(normalized_face,(350,350)))  # do prediction
            mixer.init()
            if prediction[0]==0:
                mixer.music.load('songs/neutral.mp3')
                mixer.music.play(-1,2.0)
                time.sleep(20)
                mixer.music.stop()
            if prediction[0]==1:
                mixer.music.load('songs/anger.mp3')
                mixer.music.play(-1,2.0)
                time.sleep(10)
                mixer.music.stop()
            if prediction[0]==5:
                mixer.music.load('songs/surprise.mp3')
                mixer.music.play(-1,2.0)
                time.sleep(10)
                mixer.music.stop()
            if prediction[0]==3:
                mixer.music.load('songs/happy.mp3')
                mixer.music.play(-1,2.0)
                time.sleep(10)
                mixer.music.stop()
            if prediction[0]==4:
                mixer.music.load('songs/sadness.mp3')
                mixer.music.play(-1,2.0)
                time.sleep(10)
                mixer.music.stop()
            image_to_draw = emoticons[prediction[0]]
            draw_with_alpha(webcam_image, image_to_draw, (x, y, w, h))

        cv2.imshow(window_name, webcam_image)
        read_value, webcam_image = vc.read()
        key = cv2.waitKey(update_time)

        if key == 27:  # exit on ESC
            break

    cv2.destroyWindow(window_name)



if __name__ == '__main__':
    emotions = ['neutral', 'anger', 'disgust', 'happy', 'sadness', 'surprise']
    emoticons = _load_emoticons(emotions)
    #print(len(emoticons))
    # load model
    fisher_face = cv2.face.FisherFaceRecognizer_create()
    fisher_face.read('model/emotion_detection_model1.xml')

    # use learnt model
    window_name = 'WEBCAM (press ESC to exit)'
    show_webcam_and_run(fisher_face, emoticons, window_size=(1600, 1200), window_name=window_name, update_time=8)
