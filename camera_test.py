import cv2

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: CAMERA NOT OPENING")
    exit()

print("CAMERA OPENED")
print("PRESS Q TO CLOSE")

while True:
    ret, frame = camera.read()

    if not ret:
        print("ERROR: CANNOT READ CAMERA")
        break

    frame = cv2.flip(frame, 1)

    cv2.imshow("Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

camera.release()
cv2.destroyAllWindows()