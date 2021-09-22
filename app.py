from flask import Flask, render_template, Response, request
import cv2
import json

# custom package imports
from data_ingestion.data_import_and_preprocessing import DataImport, Preprocessing
from attendance_api_functions.api_functions import API_Functions
from database_api_functions.db_api_functions import DatabaseAPI

# instatiate flask app
app = Flask(__name__, template_folder='./templates')


# making objects
camera = cv2.VideoCapture(0)  # change this if camera not working
data_import = DataImport()
preprocessing = Preprocessing()


# loading config.json
with open('config.json') as f:
    config_file = json.load(f)

#creating database object
database = DatabaseAPI(camera, "mongodb://localhost:27017/",
                       "Attendance", config_file["saved_image_folder"])


# making all the folders
data_import.make_folders(
    config_file["saved_image_folder"], config_file["attendance_folder_path"], config_file["image_path"])

# creating csv file
# attendance_file_path = data_import.make_csv_file()

#creating collection
database.make_database_collection()

# calling important functions
images, known_face_names = data_import.read_images(
    path=config_file["image_path"])
known_face_encodings = preprocessing.faceEncodings(images)
api_functions = API_Functions(
    camera, known_face_names, known_face_encodings, config_file["saved_image_folder"])



@app.route('/')
def index():
    # global attendance_file_path
    # attendance_file_path = data_import.make_csv_file()
    database.make_database_collection()
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(api_functions.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/checkin', methods=['POST'])
def checkin():
    name = request.form.get("name")
    # status = api_functions.check_in(
    #     name, attendance_file_path, save_image=True)
    status = database.check_in(name,save_image=True)
    return render_template('result.html', status='Checked In Status For {} : {} '.format(name, status))


@app.route('/checkout', methods=['POST'])
def checkout():
    name = request.form.get("name")
    # status = api_functions.check_out(
    #     name, attendance_file_path, save_image=True)
    status = database.check_out(name,save_image=True)
    return render_template('result.html', status='Checked Out Status {} : {} '.format(name, status))


@app.route('/confirm', methods=['POST'])
def confirm():
    name = api_functions.gen_name()
    if name == "Unknown": 
        return render_template('unknown.html', status="Unknown can't check in or check out")
    return render_template('mid.html', status='You are {} , Press Check In Check Out  '.format(name), name=name)


if __name__ == '__main__':
    app.run(debug=True)

camera.release()
cv2.destroyAllWindows()
