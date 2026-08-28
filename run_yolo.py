import cv2
from ultralytics import YOLO

# ==========================================
# REGISTRATION NUMBER
# ==========================================

REGISTRATION_NUMBER = "25TBY0034"


# ==========================================
# LOAD TRAINED YOLOv8n MODEL
# ==========================================

model = YOLO("runs/detect/train-2/weights/best.pt")

print("trained model loaded successfully!")
print("starting camera...")


# ==========================================
# OPEN CAMERA
# ==========================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: camera could not be opened")
    exit()


# ==========================================
# CAMERA LOOP
# ==========================================

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: could not read camera")
        break
    frame=cv2.flip(frame,1)
    # run detection
    results = model(frame, conf=0.1)

    detected = False

    for result in results:

        for box in result.boxes:

            detected = True

            # coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # confidence
            confidence = float(box.conf[0])

            # draw bounding box
            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # show registration number
        
            cv2.putText(
                frame,
                REGISTRATION_NUMBER,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

    # status
    if detected:
        cv2.putText(
            frame,
            "IDENTIFIED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            frame,
            "LOOKING FOR PERSON...",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2
        )

    # show camera
    cv2.imshow("Registration Number Recognition", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ==========================================
# CLOSE
# ==========================================

camera.release()
cv2.destroyAllWindows()

print("program stopped.")