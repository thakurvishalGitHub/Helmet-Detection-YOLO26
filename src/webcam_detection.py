import cv2
from ultralytics import YOLO

MODEL_PATH = "models/best.pt"
CONFIDENCE = 0.40

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Camera could not be opened.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame.")
        break

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        device=0,
        verbose=False
    )

    annotated_frame = results[0].plot()

    cv2.imshow(
        "Helmet Detection - Press Q to Exit",
        annotated_frame
    )

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()