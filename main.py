import cv2
import mediapipe as mp

# Initialize MediaPipe Hands and Drawing utils
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils


# Function to detect if a finger is up
def finger_status(lmList, tip_id, mcp_id):
    return lmList[tip_id][2] < lmList[mcp_id][2]


cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []
    if results.multi_hand_landmarks:
        for handlandmark in results.multi_hand_landmarks:
            for id, lm in enumerate(handlandmark.landmark):
                h, w, _ = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img, handlandmark, mpHands.HAND_CONNECTIONS)

    if lmList != []:
        fingers = []

        # Thumb
        fingers.append(lmList[4][1] > lmList[3][1])  # Right hand: thumb tip is to the right of the index finger

        # Fingers (index, middle, ring, pinky)
        for tip_id, mcp_id in zip([8, 12, 16, 20], [6, 10, 14, 18]):
            fingers.append(finger_status(lmList, tip_id, mcp_id))

        # Count fingers that are up
        totalFingers = fingers.count(True)

        # Display number of fingers
        cv2.putText(img, str(totalFingers), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 2)

        print(totalFingers)

    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
