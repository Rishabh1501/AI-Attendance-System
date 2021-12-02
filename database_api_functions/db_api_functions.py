import os
import cv2
import pymongo
from datetime import datetime


class DatabaseAPI:
    def __init__(self, camera, mongo_db_url, database_name, img_folder_path):
        self.camera = camera
        self.img_folder_path = img_folder_path
        try:
            # database name check and initialization
            client = pymongo.MongoClient(mongo_db_url)
            DBlist = client.list_database_names()
            if database_name in DBlist:
                print(f"DB: '{database_name}' exists")
            else:
                print(
                    f"DB: '{database_name}' not yet present OR no collection is present in the DB"
                )
            self.database = client[database_name]
        except pymongo.errors.ConnectionFailure as e:
            print("Exception occured while making connection")
            raise Exception(e)

    def make_database_collection(self, collection_name=None):
        if not collection_name:
            date = datetime.now().date()
            collection_name = "Attendance_" + str(date)
        collection_list = self.database.list_collection_names()
        if collection_name in collection_list:
            print(f"Collection:'{collection_name}' exists in Database")

        else:
            print(
                f"Collection:'{collection_name}' does not exists OR no documents are present in the collectionin Database"
            )
        self.collection = self.database[collection_name]

    def check_in(self, name, save_image=True):

        if not self.collection.find_one({"Name": name}):
            time_now = datetime.now()
            tStr = time_now.strftime("%H:%M:%S")
            dStr = time_now.strftime("%d/%m/%Y")
            cin = 1
            # df.loc[name] = [tStr, dStr, cin, None, None]
            # df.to_csv(attendance_file_path, index_label='Name')
            record = {
                "Name": name,
                "Time": tStr,
                "Date": dStr,
                "Check In": cin,
                "Check Out": None,
                "Check Out Time": None,
            }
            self.collection.insert_one(record)

            self.capture_frame(name,
                               check_status="check_in",
                               save_image=save_image)
            checkin_status = (
                "You Successfully Checked In at " +
                str(time_now.strftime("%H:%M:%S")) +
                ". Welcome to the Company . Have a Great Day at Work ")
            return checkin_status

        else:

            if self.collection.find_one({"Name": name, "Check Out": 1}):
                checkin_status = (
                    "You Already Checked Out at " +
                    str(self.collection.find_one({"Name": name})["Time"]) +
                    " ! See You Tomorrow :)")
            elif self.collection.find_one({"Name": name, "Check In": 1}):
                checkin_status = (
                    "You Already Checked in at " +
                    str(self.collection.find_one({"Name": name})["Time"]) +
                    " ! You can now Checked Out Only :)")
            return checkin_status

    def check_out(self, name, save_image=True):
        if not self.collection.find_one({"Name": name}):
            checkout_status = "You Have not Checked In Yet"

        else:
            if self.collection.find_one({"Name": name, "Check Out": 1}):
                checkout_status = "You Already Checked Out at " + str(
                    self.collection.find_one({"Name": name})["Check Out Time"])
            else:
                time_now = datetime.now()
                self.collection.update_one(
                    self.collection.find_one({"Name": name}),
                    {
                        "$set": {
                            "Check Out": 1,
                            "Check Out Time": time_now.strftime("%H:%M:%S"),
                        }
                    },
                )

                self.capture_frame(name,
                                   check_status="check_out",
                                   save_image=save_image)
                checkout_status = "Successfully Checked Out at " + str(
                    self.collection.find_one({"Name": name})["Check Out Time"])

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
