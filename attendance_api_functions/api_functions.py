import os
import cv2
import numpy as np
import pandas as pd
import face_recognition
from datetime import datetime


class API_Functions:

    def __init__(self, camera, known_face_names, known_face_encodings, img_folder_path):
        self.camera = camera
        self.known_face_names = known_face_names
        self.known_face_encodings = known_face_encodings
        self.img_folder_path = img_folder_path

    def check_in(self, name, attendance_file_path, save_image=True):

        df = pd.read_csv(attendance_file_path, index_col='Name')
        if name not in df.index:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            cin = 1
            df.loc[name] = [tStr, dStr, cin, None, None]
            df.to_csv(attendance_file_path, index_label='Name')
            self.capture_frame(name, check_status="check_in",
                               save_image=save_image)
            checkin_status = "You Successfully Checked In at " + \
                str(time_now.strftime('%H:%M:%S')) + \
                ". Welcome to the Company . Have a Great Day at Work "
            return checkin_status

        else:
            if(df['Check Out'][name] == 1):
                checkin_status = "You Already Checked Out at " + \
                    str(df['Check Out Time'][name]) + " ! See You Tomorrow :)"
            elif(df['Check In'][name] == 1):
                checkin_status = "You Already Checked in at " + \
                    str(df['Time'][name]) + \
                    " ! You can now Checked Out Only :)"
            return checkin_status

    def check_out(self, name, attendance_file_path, save_image=True):
        df = pd.read_csv(attendance_file_path, index_col='Name')
        if name not in df.index:
            checkout_status = "You Have not Checked In Yet"

        else:
            if(df['Check Out'][name] == 1):
                checkout_status = "You Already Checked Out at " + \
                    str(df['Check Out Time'][name])
            else:
                df['Check Out'][name] = 1
                time_now = datetime.now()
                df['Check Out Time'][name] = time_now.strftime('%H:%M:%S')
                self.capture_frame(
                    name, check_status="check_out", save_image=save_image)
                checkout_status = "Successfully Checked Out at " + \
                    str(df['Check Out Time'][name])

                df.to_csv(attendance_file_path, index_label='Name')

        return checkout_status

    def capture_frame(self, name, check_status, save_image=True):
        if self.img_folder_path and save_image:
            success, frame = self.camera.read()
            path = f"{self.img_folder_path}//{name}"
            if name not in os.listdir(self.img_folder_path):
                os.mkdir(path)
            img_path = f"{path}//{name}_{str(datetime.now().date())}_time-{str(datetime.now().time().strftime('%H-%M-%S'))}_{check_status}.jpg"
            check = cv2.imwrite(img_path, frame)
            if check:
                print("Image Saved Successfully")

    def gen_frames(self):
        while True:
            success, frame = self.camera.read()  # read the camera frame
            if not success:
                break
            else:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Only process every other frame of video to save time

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(
                    rgb_small_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding, tolerance=0.5)
                    name = "Unknown"
                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

                    face_names.append(name)

                # Display the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top),
                                  (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(frame, (left, bottom - 35),
                                  (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (left + 6, bottom - 6),
                                font, 1.0, (255, 255, 255), 1)

                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    def gen_name(self):
        while True:
            success, frame = self.camera.read()  # read the camera frame
            name = "Unknown"  # changed
            if not success:
                break
            else:
                # Resize frame of video to 1/4 size for faster face recognition processing
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = small_frame[:, :, ::-1]

                # Only process every other frame of video to save time

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(
                    rgb_small_frame)
                face_encodings = face_recognition.face_encodings(
                    rgb_small_frame, face_locations)
                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(
                        self.known_face_encodings, face_encoding, tolerance=0.5)
                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(
                        self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]

                    face_names.append(name)

                return name
