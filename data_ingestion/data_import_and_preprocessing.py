# library imports
import os
import cv2
import pandas as pd
import face_recognition
from datetime import datetime


class DataImport:
    def make_folders(self,
                     saved_image_folder=None,
                     attendance_folder_path=None,
                     image_folder=None):

        if not saved_image_folder:
            if "saved_images" not in os.listdir(os.getcwd()):
                os.mkdir(os.path.join(os.getcwd(), "saved_images"))
        else:
            if saved_image_folder not in os.listdir(os.getcwd()):
                os.mkdir(os.path.join(os.getcwd(), saved_image_folder))

        if not attendance_folder_path:
            if "Attendance" not in os.listdir(os.getcwd()):
                os.mkdir(os.path.join(os.getcwd(), "Attendance"))
                self.attendance_folder_path = os.path.join(
                    os.getcwd(), "Attendance")
        else:
            if attendance_folder_path not in os.listdir(os.getcwd()):
                os.mkdir(os.path.join(os.getcwd(), attendance_folder_path))
            self.attendance_folder_path = os.path.join(os.getcwd(),
                                                       attendance_folder_path)

        if not image_folder:
            if "images" not in os.listdir(os.getcwd()):
                os.mkdir(os.path.join(os.getcwd(), "images"))
        else:
            if image_folder not in os.listdir(os.getcwd()):
                os.mkdir(os.path.join(os.getcwd(), image_folder))

    def make_csv_file(self):
        date = datetime.now().date()
        attendance_file_path = os.path.join(self.attendance_folder_path,
                                            "Attendance_" + str(date) + ".csv")
        exists = os.path.isfile(attendance_file_path)
        if exists:
            print("Attendance File Present")
        else:
            try:
                with open(attendance_file_path, "a+"):
                    data = {
                        "Name": [],
                        "Time": [],
                        "Date": [],
                        "Check In": [],
                        "Check Out": [],
                        "Check Out Time": []
                    }
                    df = pd.DataFrame(
                        data,
                        columns=[
                            "Name", "Time", "Date", "Check In", "Check Out",
                            "Check Out Time"
                        ],
                    )  # create DataFrame
                    df.set_index("Name", inplace=True)
                    df.to_csv(attendance_file_path, sep=",", header=True)
                    print("Attendance File Created")
            except Exception as e:
                print("ERROR IN CREATING Attendance File!!!")
                raise Exception(e)
        return attendance_file_path

    def read_images(self, path=None):
        if not path:
            path = "images"
        images = []
        known_face_names = []
        image_list = os.listdir(path)
        print(f"Images Present in {path} are: {image_list}")
        for img in image_list:
            current_Img = cv2.imread(os.path.join(path, img))
            images.append(current_Img)
            known_face_names.append(os.path.splitext(img)[0])
        print(f"Names Extracted from Images: {known_face_names}")

        return images, known_face_names


class Preprocessing:
    def faceEncodings(self, images):
        encodeList = []
        for img in images:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            encode = face_recognition.face_encodings(img)[0]
            encodeList.append(encode)
        return encodeList
