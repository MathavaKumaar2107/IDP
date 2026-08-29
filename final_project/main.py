import cv2
import face_recognition
import os
import csv
import time
from datetime import datetime


# =====================================================
# SETTINGS
# =====================================================

ABSENCE_TIME = 5

FACE_TOLERANCE = 0.50


# =====================================================
# LOAD FACULTY
# =====================================================

faculty_encodings = []
faculty_names = []

for filename in os.listdir("faculty"):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        path = os.path.join(
            "faculty",
            filename
        )

        image = face_recognition.load_image_file(path)

        encodings = face_recognition.face_encodings(
            image
        )

        if len(encodings) == 1:

            faculty_encodings.append(
                encodings[0]
            )

            faculty_names.append(
                os.path.splitext(filename)[0]
            )


# =====================================================
# LOAD STUDENTS
# =====================================================

student_encodings = []
student_registers = []

for filename in os.listdir("students"):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png")
    ):

        path = os.path.join(
            "students",
            filename
        )

        image = face_recognition.load_image_file(path)

        encodings = face_recognition.face_encodings(
            image
        )

        if len(encodings) == 1:

            student_encodings.append(
                encodings[0]
            )

            student_registers.append(
                os.path.splitext(filename)[0]
            )


# =====================================================
# CHECK DATA
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
# STUDENT STATUS
# =====================================================

student_status = {}

for register_no in student_registers:

    student_status[register_no] = {

        "status": "NOT DETECTED",

        "first_seen": None,

        "last_seen": None
    }


# =====================================================
# SESSION
# =====================================================

session_started = False


# =====================================================
# CAMERA
# =====================================================

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print("ERROR: Camera could not be opened.")

    exit()


print("Waiting for faculty...")
print("Show faculty face to start attendance.")
print("Press Q to quit.")
print()


# =====================================================
# MAIN LOOP
# =====================================================

while True:

    ret, frame = camera.read()

    if not ret:
        break

    # Mirror camera

    frame = cv2.flip(frame, 1)


    # Smaller frame for faster processing

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
    # FACULTY VERIFICATION
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

                faculty_verified = True

                print()
                print("======================================")
                print("FACULTY VERIFIED")
                print("Faculty:", faculty_name)
                print("ATTENDANCE SESSION STARTED")
                print("======================================")
                print()

                break


        # Screen message

        if faculty_verified:

            cv2.putText(
                frame,
                "FACULTY VERIFIED",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                "ATTENDANCE STARTED",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "SHOW FACULTY FACE",
                (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )


    # =================================================
    # STUDENT ATTENDANCE
    # =================================================

    else:

        current_time = time.time()


        locations = face_recognition.face_locations(
            rgb_small
        )

        encodings = face_recognition.face_encodings(
            rgb_small,
            locations
        )


        # Students detected in THIS frame

        detected_students = set()


        # =================================================
        # IDENTIFY EACH FACE
        # =================================================

        for encoding, location in zip(
            encodings,
            locations
        ):

            matches = face_recognition.compare_faces(
                student_encodings,
                encoding,
                tolerance=FACE_TOLERANCE
            )


            register_no = "UNKNOWN"


            if True in matches:

                index = matches.index(True)

                register_no = student_registers[index]

                detected_students.add(
                    register_no
                )


                # -----------------------------------------
                # FIRST DETECTION
                # -----------------------------------------

                if student_status[
                    register_no
                ]["first_seen"] is None:

                    student_status[
                        register_no
                    ]["first_seen"] = current_time

                    student_status[
                        register_no
                    ]["status"] = "PRESENT"

                    print(
                        register_no,
                        "-> PRESENT"
                    )


                # -----------------------------------------
                # UPDATE LAST SEEN
                # -----------------------------------------

                student_status[
                    register_no
                ]["last_seen"] = current_time

                student_status[
                    register_no
                ]["status"] = "PRESENT"


            # =================================================
            # FACE COORDINATES
            # =================================================

            top, right, bottom, left = location


            # Convert small-frame coordinates
            # back to original frame

            top *= 4
            right *= 4
            bottom *= 4
            left *= 4


            # =================================================
            # DRAW BOX
            # =================================================

            if register_no == "UNKNOWN":

                box_color = (0, 0, 255)

            else:

                box_color = (0, 255, 0)


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
        # CHECK 5 SECOND ABSENCE
        # =================================================

        for register_no in student_registers:

            last_seen = student_status[
                register_no
            ]["last_seen"]


            # -----------------------------------------
            # NEVER SEEN
            # -----------------------------------------

            if last_seen is None:

                student_status[
                    register_no
                ]["status"] = "ABSENT"


            # -----------------------------------------
            # PREVIOUSLY SEEN
            # -----------------------------------------

            else:

                time_missing = (
                    current_time - last_seen
                )


                if time_missing >= ABSENCE_TIME:

                    if student_status[
                        register_no
                    ]["status"] == "PRESENT":

                        print(
                            register_no,
                            "-> ABSENT / LEFT CLASS"
                        )


                    student_status[
                        register_no
                    ]["status"] = "ABSENT"


        # =================================================
        # DISPLAY STUDENT STATUS
        # =================================================

        y = 180


        for register_no in student_registers:

            status = student_status[
                register_no
            ]["status"]


            text = (
                f"{register_no} : {status}"
            )


            if status == "PRESENT":

                text_color = (0, 255, 0)

            else:

                text_color = (0, 0, 255)


            cv2.putText(
                frame,
                text,
                (30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                text_color,
                2
            )


            y += 30


        # =================================================
        # ATTENDANCE COUNT
        # =================================================

        present_count = 0


        for register_no in student_registers:

            if student_status[
                register_no
            ]["status"] == "PRESENT":

                present_count += 1


        cv2.putText(
            frame,
            f"PRESENT: {present_count}/{len(student_registers)}",
            (30, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        cv2.putText(
            frame,
            "ATTENDANCE ACTIVE",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )


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
        "status",
        "time"
    ])


    for register_no in student_registers:

        writer.writerow([

            register_no,

            student_status[
                register_no
            ]["status"],

            datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S"
            )
        ])


# =====================================================
# FINAL OUTPUT
# =====================================================

print()
print("======================================")
print("FINAL ATTENDANCE")
print("======================================")


for register_no in student_registers:

    print(
        register_no,
        "->",
        student_status[
            register_no
        ]["status"]
    )


print()
print("Attendance saved to attendance.csv")