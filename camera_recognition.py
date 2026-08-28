import cv2
import face_recognition

# ---------------------------------------
# STUDENT DETAILS
# ---------------------------------------

REGISTER_NUMBER = "25BYB0034"

REFERENCE_IMAGE = "dataset/images/train/photo.jpg"


# ---------------------------------------
# LOAD REFERENCE PHOTO
# ---------------------------------------

reference_image = face_recognition.load_image_file(
    REFERENCE_IMAGE
)

reference_faces = face_recognition.face_encodings(
    reference_image
)

if len(reference_faces) == 0:
    print("ERROR: No face found in reference photo.")
    exit()

if len(reference_faces) > 1:
    print("ERROR: Reference photo contains more than one face.")
    exit()

reference_encoding = reference_faces[0]

print("Reference face loaded successfully.")
print("Register number:", REGISTER_NUMBER)


# ---------------------------------------
# OPEN CAMERA
# ---------------------------------------

camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("ERROR: Camera could not be opened.")
    exit()

print("Camera started.")
print("Press Q to quit.")


# ---------------------------------------
# CAMERA LOOP
# ---------------------------------------

while True:

    ret, frame = camera.read()

    if not ret:
        print("ERROR: Could not read camera.")
        break

    # Mirror camera
    frame = cv2.flip(frame, 1)

    # Convert BGR → RGB
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Find faces
    face_locations = face_recognition.face_locations(
        rgb_frame
    )

    face_encodings = face_recognition.face_encodings(
        rgb_frame,
        face_locations
    )

    # -----------------------------------
    # CHECK EACH FACE
    # -----------------------------------

    for face_encoding, location in zip(
        face_encodings,
        face_locations
    ):

        # Compare with registered face
        distance = face_recognition.face_distance(
            [reference_encoding],
            face_encoding
        )[0]

        # Lower distance = better match
        is_match = distance < 0.50

        top, right, bottom, left = location

        if is_match:

            label = f"{REGISTER_NUMBER}"

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                label,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

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

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                (0, 0, 255),
                2
            )

            cv2.putText(
                frame,
                "UNKNOWN",
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

    # -----------------------------------
    # SHOW CAMERA
    # -----------------------------------

    cv2.imshow(
        "Student Identification",
        frame
    )

    # Q = quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


# ---------------------------------------
# CLOSE
# ---------------------------------------

camera.release()
cv2.destroyAllWindows()