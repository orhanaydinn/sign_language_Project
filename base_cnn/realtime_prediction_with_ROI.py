import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import deque

model = load_model("best_model.keras")
label_map = {
    0: "zero", 1: "one", 2: "two", 3: "three", 4: "four",
    5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine"
}

IMG_SIZE = 64
CONFIDENCE_THRESHOLD = 0.80
VOTE_HISTORY = 15
history = deque(maxlen=VOTE_HISTORY)


roi_selected = False
roi_start = (0, 0)
roi_end = (0, 0)

def preprocess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
    normalized = resized / 255.0
    return normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)

# 🖱️ Mouse callback
def draw_roi(event, x, y, flags, param):
    global roi_start, roi_end, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (x, y)
        roi_selected = True
        print(f"ROI selected: {roi_start} → {roi_end}")

cap = cv2.VideoCapture(0)
cv2.namedWindow("Live Prediction")
cv2.setMouseCallback("Live Prediction", draw_roi)

print("🖱️ Draw Roi with mouse")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    display_frame = frame.copy()

    # ROI seçildiyse tahmin yap
    if roi_selected:
        x1, y1 = roi_start
        x2, y2 = roi_end
        x, y, w, h = min(x1,x2), min(y1,y2), abs(x2-x1), abs(y2-y1)
        roi = frame[y:y+h, x:x+w]
        processed = preprocess(roi)

        prediction = model.predict(processed, verbose=0)
        class_id = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        if confidence > CONFIDENCE_THRESHOLD:
            history.append(class_id)
            final_pred = max(set(history), key=history.count)
            text = f"{label_map[final_pred]} ({confidence*100:.1f}%)"
            color = (0, 255, 0)
        else:
            text = "Wrong!"
            color = (0, 0, 255)

        # ROI kutusunu çiz
        cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(display_frame, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    else:
        cv2.putText(display_frame, "Draw ROI with mouse", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.imshow("Live Prediction", display_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
