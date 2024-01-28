import cv2
import dlib
import mediapipe as mp
import keyboard
import time
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import webbrowser


# Initialize Dlib face detector
detector = dlib.get_frontal_face_detector()
predictor_path = "file/shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Initialize MediaPipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

recognizer = sr.Recognizer()

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera")

current_time = 0
speech_active = True
last_exec_time = 0
last_reset_time = 0

websites = {}

with open("file/website_url.txt") as file:
    for line in file:
        name, url = line.strip().split(',')
        websites[name] = url

def detect_face_and_hand():
    global current_time
    global last_exec_time
    global last_reset_time
    last_exec_time = last_exec_time or 0
    last_reset_time = last_reset_time or 0
    execute_once_flag = False
    execute_once_flag_2 = False
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Face detection using Dlib
    faces = detector(gray)
    for face in faces:
        landmark = predictor(gray, face)
        for i in range(68):
            x, y = landmark.part(i).x, landmark.part(i).y
            cv2.circle(frame, (x, y), 1, (0, 255, 0), 1)

    # Hand detection using MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        middle_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

        thumb_x, thumb_y = int(thumb_tip.x * frame.shape[1]), int(thumb_tip.y * frame.shape[0])
        index_x, index_y = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0])
        middle_x, middle_y = int(middle_tip.x * frame.shape[1]), int(middle_tip.y * frame.shape[0])
        pinky_tip_x, pinky_tip_y = int(pinky_tip.x * frame.shape[1]), int(pinky_tip.y * frame.shape[0])

        if index_x > thumb_x:
            current_time = time.time()

            if current_time - last_exec_time > 3 and not execute_once_flag:
                keyboard.press_and_release('win+ctrl+right')
                execute_once_flag = True

                last_exec_time = current_time

        elif index_x < thumb_x:
            current_time = time.time()
            if current_time - last_exec_time > 3 and not execute_once_flag_2:
                keyboard.press_and_release('win+ctrl+left')
                execute_once_flag_2 = True
                last_exec_time = current_time

        for hand_landmarks in results.multi_hand_landmarks:
            for j, point in enumerate(hand_landmarks.landmark):
                x, y, z = int(point.x * frame.shape[1]), int(point.y * frame.shape[0]), int(point.z * frame.shape[1])
                cv2.circle(frame, (x, y), 1, (255, 0, 0), 1)

        mp_drawing = mp.solutions.drawing_utils
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        return False

    current_time = time.time()
    if current_time - last_reset_time > 5:
        execute_once_flag = False
        last_reset_time = current_time

    return True



def detect_speech():
    global current_time
    with sr.Microphone() as source:
        print("say something")
        recognizer.adjust_for_ambient_noise(source)
        audio = recognizer.listen(source)

    try:
        text = recognizer.recognize_google(audio).lower()
        print("you said", text)

        if "open" in text and any(website in text for website in websites):
            for website_name, website_url in websites.items():
                if website_name in text:
                    webbrowser.open(website_url)
                    print(f"Opening {website_name}")
                    break
        else:
            print("Command not recognized or website not found in dictionary.")

        last_recognition_time = current_time

        return text # Return the detected speech text
    except sr.UnknownValueError:
        print("could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"error with service; {e}")
        return ""


while True:
    if speech_active:
        detected_speech = detect_speech()

        if "detect face" in detected_speech.lower():
            speech_active = False
            while detect_face_and_hand():
                pass  # You can leave this empty if you don't need any specific action
            speech_active = True

        # Add more conditions for other commands if needed

    # If you want to add a command to exit the loop:
    # if "exit" in detected_speech.lower():
    #     break

cap.release()
cv2.destroyAllWindows()


cap.release()
cv2.destroyAllWindows()
