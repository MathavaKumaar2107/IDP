import cv2
import face_recognition
import os
import csv
import time
from datetime import datetime


# =====================================================
# SETTINGS
# =====================================================

SESSION_TIME = 20       # 20 seconds for testing
LEFT_TIME = 5           # 5 seconds not seen = left
FACE_TOLERANCE = 0.50


# =====================================================
# FOLDERS
# =====================================================

FACULTY_FOLDER = "faculty"
STUDENT_FOLDER = "students"


# =====================================================
# LOAD FACULTY
# =====================================================

faculty_encodings = []
faculty_names = []

if not os.path.exists(FACULTY_FOLDER):
    print("ERROR: faculty folder not found.")
    exit()

for filename in os.listdir(FACULTY_FOLDER):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        path = os.path.join(
            FACULTY_FOLDER,
            filename
        )

        image = face_recognition.load_image_file(path)

        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 1:

            faculty_encodings.append(encodings[0])

            faculty_names.append(
                os.path.splitext(filename)[0]
            )


# =====================================================
# LOAD STUDENTS
# =====================================================

student_encodings = []
student_registers = []

if not os.path.exists(STUDENT_FOLDER):
    print("ERROR: students folder not found.")
    exit()

for filename in os.listdir(STUDENT_FOLDER):

    if filename.lower().endswith((".jpg", ".jpeg", ".png")):

        path = os.path.join(
            STUDENT_FOLDER,
            filename
        )

        image = face_recognition.load_image_file(path)

        encodings = face_recognition.face_encodings(image)

        if len(encodings) == 1:

            student_encodings.append(encodings[0])

            student_registers.append(
                os.path.splitext(filename)[0]
            )


# =====================================================
# DISPLAY LOADED DATA
# =====================================================

print()
print("======================================")
print("CLASSROOM ATTENDANCE SYSTEM")
print("======================================")

print("Faculty loaded:", faculty_names)

print("Students loaded:", student_registers)

print()


if len(faculty_encodings) == 0:

    print("ERROR: No faculty face found.")

    exit()


if len(student_encodings) == 0:

    print("ERROR: No student faces found.")

    exit()


# =====================================================
# STUDENT INFORMATION
# =====================================================

student_status = {}

for register_no in student_registers:

    student_status[register_no] = {

        # Did the student give biometric?
        "biometric": False,

        # Current status
        "status": "ABSENT",

        # Last time CCTV saw student
        "last_seen": None
    }


# =====================================================
# SESSION VARIABLES
# =====================================================

session_started = False

session_start_time = None


# =====================================================
# OPEN WEBCAM
# =====================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


print("Waiting for faculty...")
print("Faculty must show face.")
print("Press Q to quit.")
print()


# =====================================================
# MAIN LOOP
# =====================================================

while True:

    ret, frame = camera.read()

    if not ret:

        print("ERROR: Could not read camera.")

        break


    # Mirror webcam

    frame = cv2.flip(frame, 1)


    # Resize for faster processing

    small_frame = cv2.resize(
        frame,
        (0, 0),
        fx=0.25,
        fy=0.25
    )


    rgb_small = cv2.cvtColor(
        small_frame,
        cv2.COLOR_BGR2RGB
    )


    # =================================================
    # STEP 1: FACULTY VERIFICATION
    # =================================================

    if not session_started:

        locations = face_recognition.face_locations(
            rgb_small
        )

        encodings = face_recognition.face_encodings(
            rgb_small,
            locations
        )


        faculty_verified = False


        for encoding in encodings:

            matches = face_recognition.compare_faces(
                faculty_encodings,
                encoding,
                tolerance=FACE_TOLERANCE
            )


            if True in matches:

                index = matches.index(True)

                faculty_name = faculty_names[index]

                session_started = True

                session_start_time = time.time()

                faculty_verified = True


                print()
                print("======================================")
                print("FACULTY VERIFIED")
                print("Faculty:", faculty_name)
                print("ATTENDANCE SESSION STARTED")
                print("SESSION TIME:", SESSION_TIME, "SECONDS")
                print("======================================")
                print()

                break


        if not faculty_verified:

            cv2.putText(
                frame,
                "SHOW FACULTY FACE",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )


        else:

            cv2.putText(
                frame,
                "FACULTY VERIFIED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )


    # =================================================
    # STEP 2: ATTENDANCE SESSION
    # =================================================

    else:

        current_time = time.time()

        elapsed_time = (
            current_time - session_start_time
        )

        remaining_time = max(
            0,
            SESSION_TIME - elapsed_time
        )


        # =================================================
        # FIND FACES
        # =================================================

        locations = face_recognition.face_locations(
            rgb_small
        )

        encodings = face_recognition.face_encodings(
            rgb_small,
            locations
        )


        # =================================================
        # PROCESS EACH FACE
        # =================================================

        for encoding, location in zip(
            encodings,
            locations
        ):

            # ---------------------------------------------
            # COMPARE WITH STUDENTS
            # ---------------------------------------------

            matches = face_recognition.compare_faces(
                student_encodings,
                encoding,
                tolerance=FACE_TOLERANCE
            )


            register_no = "UNKNOWN"


            if True in matches:

                index = matches.index(True)

                register_no = student_registers[index]


                # =================================================
                # STUDENT BIOMETRIC VERIFICATION
                # =================================================

                if not student_status[
                    register_no
                ]["biometric"]:

                    student_status[
                        register_no
                    ]["biometric"] = True

                    student_status[
                        register_no
                    ]["status"] = "PRESENT"

                    print(
                        register_no,
                        "-> BIOMETRIC VERIFIED -> PRESENT"
                    )


                # =================================================
                # CCTV SEES VERIFIED STUDENT
                # =================================================

                student_status[
                    register_no
                ]["last_seen"] = current_time


                # If student had left and came back

                if student_status[
                    register_no
                ]["status"] == "LEFT THE CLASS":

                    student_status[
                        register_no
                    ]["status"] = "PRESENT"

                    print(
                        register_no,
                        "-> RETURNED TO CLASS"
                    )


            # =================================================
            # FACE BOX
            # =================================================

            top, right, bottom, left = location


            # Convert coordinates back
            # to original frame size

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4


            # =================================================
            # BOX COLOR
            # =================================================

            if register_no == "UNKNOWN":

                box_color = (0, 0, 255)

            else:

                box_color = (0, 255, 0)


            # =================================================
            # DRAW BOX
            # =================================================

            cv2.rectangle(
                frame,
                (left, top),
                (right, bottom),
                box_color,
                2
            )


            # =================================================
            # SHOW REGISTER NUMBER
            # =================================================

            cv2.putText(
                frame,
                register_no,
                (left, top - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                box_color,
                2
            )


        # =================================================
        # CHECK STUDENT STATUS
        # =================================================

        for register_no in student_registers:

            student = student_status[
                register_no
            ]


            # =================================================
            # NEVER GAVE BIOMETRIC
            # =================================================

            if not student["biometric"]:

                student["status"] = "ABSENT"

                continue


            # =================================================
            # BIOMETRIC GIVEN
            # =================================================

            last_seen = student["last_seen"]


            if last_seen is not None:

                missing_time = (
                    current_time - last_seen
                )


                # =================================================
                # LEFT CLASS
                # =================================================

                if missing_time >= LEFT_TIME:

                    if student["status"] == "PRESENT":

                        student["status"] = "LEFT THE CLASS"

                        print(
                            register_no,
                            "-> LEFT THE CLASS"
                        )


        # =================================================
        # TIMER
        # =================================================

        cv2.putText(
            frame,
            "ATTENDANCE ACTIVE",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            f"TIME LEFT: {int(remaining_time)}s",
            (30, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


        # =================================================
        # DISPLAY STUDENT STATUS
        # =================================================

        y = 125


        for register_no in student_registers:

            status = student_status[
                register_no
            ]["status"]


            text = (
                f"{register_no} : {status}"
            )


            if status == "PRESENT":

                text_color = (0, 255, 0)

            elif status == "LEFT THE CLASS":

                text_color = (0, 165, 255)

            else:

                text_color = (0, 0, 255)


            cv2.putText(
                frame,
                text,
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                text_color,
                2
            )


            y += 30


        # =================================================
        # END SESSION
        # =================================================

        if elapsed_time >= SESSION_TIME:

            print()
            print("======================================")
            print("ATTENDANCE SESSION ENDED")
            print("======================================")

            break


    # =================================================
    # SHOW CAMERA
    # =================================================

    cv2.imshow(
        "Classroom Attendance System",
        frame
    )


    # =================================================
    # QUIT
    # =================================================

    if cv2.waitKey(1) & 0xFF == ord("q"):

        print("Program stopped by user.")

        break


# =====================================================
# CLOSE CAMERA
# =====================================================

camera.release()

cv2.destroyAllWindows()


# =====================================================
# SAVE ATTENDANCE
# =====================================================

with open(
    "attendance.csv",
    "w",
    newline=""
) as file:

    writer = csv.writer(file)


    writer.writerow([
        "register_no",
        "biometric",
        "status",
        "time"
    ])


    for register_no in student_registers:

        student = student_status[
            register_no
        ]


        writer.writerow([

            register_no,

            "YES"
            if student["biometric"]
            else "NO",

            student["status"],

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ])


# =====================================================
# FINAL RESULT
# =====================================================

print()
print("======================================")
print("FINAL ATTENDANCE")
print("======================================")


for register_no in student_registers:

    student = student_status[
        register_no
    ]


    print(
        register_no,
        "->",
        student["status"]
    )


print()
print("Attendance saved to attendance.csv")
print("======================================")