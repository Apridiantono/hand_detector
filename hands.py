import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
import threading

# Fungsi putar suara di thread terpisah
def play_sound(path):
    try:
        threading.Thread(target=playsound, args=(path,)).start()
    except Exception as e:
        print("Gagal memutar suara:", e)

# Fungsi overlay gambar PNG transparan
def overlay_image_alpha(img, img_overlay, pos):
    x, y = pos
    h, w = img_overlay.shape[:2]
    if x < 0 or y < 0 or y + h > img.shape[0] or x + w > img.shape[1]:
        return
    alpha_overlay = img_overlay[:, :, 3] / 255.0
    alpha_background = 1.0 - alpha_overlay

    for c in range(3):
        img[y:y + h, x:x + w, c] = (
            alpha_overlay * img_overlay[:, :, c] +
            alpha_background * img[y:y + h, x:x + w, c]
        )

# Load template PNG RGBA
template = cv2.imread('hand_template.png', cv2.IMREAD_UNCHANGED)
if template is None or template.shape[2] != 4:
    print("Gagal memuat template PNG dengan alpha channel.")
    exit()
template_h, template_w = template.shape[:2]

# Inisialisasi MediaPipe
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Buka kamera
cap = cv2.VideoCapture(0)
cv2.namedWindow("Hand Scanner", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Hand Scanner", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

video_playing = False
sound_played = False

# Loop utama
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    center_x, center_y = w // 2 - template_w // 2, h // 2 - template_h // 2

    # Tampilkan template tangan di tengah
    overlay_image_alpha(frame, template, (center_x, center_y))

    # Deteksi tangan
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min, x_max = int(min(x_coords) * w), int(max(x_coords) * w)
            y_min, y_max = int(min(y_coords) * h), int(max(y_coords) * h)

            # Jika posisi tangan sesuai dengan template
            if (center_x < x_min < center_x + template_w and
                center_y < y_min < center_y + template_h and
                center_x < x_max < center_x + template_w and
                center_y < y_max < center_y + template_h):

                if not video_playing:
                    video_playing = True
                    sound_played = True
                    play_sound('access_granted.mp3')

                    # Mainkan video
                    video = cv2.VideoCapture('vid.mp4')
                    while video.isOpened():
                        ret_vid, frame_vid = video.read()
                        if not ret_vid:
                            break
                        cv2.imshow('Video', frame_vid)
                        if cv2.waitKey(30) & 0xFF == 27:
                            break
                    video.release()
                    cv2.destroyWindow('Video')
            else:
                if not sound_played:
                    sound_played = True
                    play_sound('access_denied.mp3')
    else:
        # Reset jika tidak ada tangan
        video_playing = False
        sound_played = False

    # Tampilkan frame fullscreen
    cv2.imshow('Hand Scanner', frame)
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Bersih-bersih
cap.release()
cv2.destroyAllWindows()
