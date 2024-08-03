import pickle
import cv2
import mediapipe as mp
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
import time

# Load the model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E', 5:'F', 6:'G', 7:'H', 8:'I', 9:'J', 10:'K', 11:'L', 12:'M', 13:'N', 14:'O', 15:'P', 16:'R', 17:'S', 18:'T', 19:'U', 20:'V', 21:'W', 22:'X', 23:'Y', 24:'Z', 25:'Space'}

# Initialize the GUI window
window = tk.Tk()
window.title("Sign Language Recognition")

# Create a label to display the camera feed
label = tk.Label(window)
label.pack()

# Create a label to display the predicted character
prediction_label = tk.Label(window, text="", font=("Helvetica", 24))
prediction_label.pack()

# Create a label to display the formed sentence
sentence_label = tk.Label(window, text="", font=("Helvetica", 24))
sentence_label.pack()

# Button to clear the sentence
def clear_sentence():
    global sentence
    sentence = []
    sentence_label.config(text="")

clear_button = tk.Button(window, text="Clear", command=clear_sentence)
clear_button.pack()

# Initialize the sentence
sentence = []

# Initialize variables to track the last detected character and time
last_detected_char = None
last_detected_time = None
stability_threshold = 3  # seconds

cap = cv2.VideoCapture(0)

def update_frame():
    global last_detected_char, last_detected_time

    ret, frame = cap.read()
    if not ret:
        return

    data_aux = []
    x_ = []
    y_ = []

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame_rgb,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        predicted_character = labels_dict[int(prediction[0])]

        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame_rgb, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3, cv2.LINE_AA)
        prediction_label.config(text=predicted_character)

        current_time = time.time()
        
        if predicted_character == last_detected_char:
            # Check if the character has been stable for the threshold duration
            if current_time - last_detected_time >= stability_threshold:
                if predicted_character == 'Space':
                    sentence.append(' ')
                else:
                    sentence.append(predicted_character)
                sentence_label.config(text=''.join(sentence))
                last_detected_char = None  # Reset after appending
        else:
            # Update the last detected character and time
            last_detected_char = predicted_character
            last_detected_time = current_time

    img = Image.fromarray(frame_rgb)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)
    window.after(10, update_frame)

update_frame()
window.mainloop()

cap.release()
cv2.destroyAllWindows()