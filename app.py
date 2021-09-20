from flask import Flask, render_template, Response, request
import cv2
import json

# custom package imports
from data_ingestion.data_import_and_preprocessing import DataImport, Preprocessing
from attendance_api_functions.api_functions import API_Functions

# instatiate flask app
app = Flask(__name__, template_folder='./templates')

# making objects
camera = cv2.VideoCapture(0)  # change this if camera not working
data_import = DataImport()
preprocessing = Preprocessing()

# loading config.json
with open('config.json') as f:
    config_file = json.load(f)

images, known_face_names = data_import.read_images(
    path=config_file["image_path"])
known_face_encodings = preprocessing.faceEncodings(images)
api_functions = API_Functions(
    camera, known_face_names, known_face_encodings, config_file["saved_image_folder"])


#global variables
attendance_file_path = None


# Initialize some variables
# face_locations = []
# face_encodings = []
# face_names = []
# process_this_frame = True
# currentname = "U"


# def gen_frames():
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         if not success:
#             break
#         else:
#             # Resize frame of video to 1/4 size for faster face recognition processing
#             small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#             # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#             rgb_small_frame = small_frame[:, :, ::-1]

#             # Only process every other frame of video to save time

#             # Find all the faces and face encodings in the current frame of video
#             face_locations = face_recognition.face_locations(rgb_small_frame)
#             face_encodings = face_recognition.face_encodings(
#                 rgb_small_frame, face_locations)
#             face_names = []
#             for face_encoding in face_encodings:
#                 # See if the face is a match for the known face(s)
#                 matches = face_recognition.compare_faces(
#                     known_face_encodings, face_encoding)
#                 name = "Unknown"
#                 # Or instead, use the known face with the smallest distance to the new face
#                 face_distances = face_recognition.face_distance(
#                     known_face_encodings, face_encoding)
#                 best_match_index = np.argmin(face_distances)
#                 if matches[best_match_index]:
#                     name = known_face_names[best_match_index]

#                 face_names.append(name)
#                 # attendance(name)
#                 currentname = name

#             # Display the results
#             for (top, right, bottom, left), name in zip(face_locations, face_names):
#                 # Scale back up face locations since the frame we detected in was scaled to 1/4 size
#                 top *= 4
#                 right *= 4
#                 bottom *= 4
#                 left *= 4

#                 # Draw a box around the face
#                 cv2.rectangle(frame, (left, top),
#                               (right, bottom), (0, 0, 255), 2)

#                 # Draw a label with a name below the face
#                 cv2.rectangle(frame, (left, bottom - 35),
#                               (right, bottom), (0, 0, 255), cv2.FILLED)
#                 font = cv2.FONT_HERSHEY_DUPLEX
#                 cv2.putText(frame, name, (left + 6, bottom - 6),
#                             font, 1.0, (255, 255, 255), 1)

#             ret, buffer = cv2.imencode('.jpg', frame)
#             frame = buffer.tobytes()
#             yield (b'--frame\r\n'
#                    b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


# def gen_name():
#     while True:
#         success, frame = camera.read()  # read the camera frame
#         name = "Unknown"  # changed
#         if not success:
#             break
#         else:
#             # Resize frame of video to 1/4 size for faster face recognition processing
#             small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
#             # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
#             rgb_small_frame = small_frame[:, :, ::-1]

#             # Only process every other frame of video to save time

#             # Find all the faces and face encodings in the current frame of video
#             face_locations = face_recognition.face_locations(rgb_small_frame)
#             face_encodings = face_recognition.face_encodings(
#                 rgb_small_frame, face_locations)
#             face_names = []
#             for face_encoding in face_encodings:
#                 # See if the face is a match for the known face(s)
#                 matches = face_recognition.compare_faces(
#                     known_face_encodings, face_encoding)
#                 # Or instead, use the known face with the smallest distance to the new face
#                 face_distances = face_recognition.face_distance(
#                     known_face_encodings, face_encoding)
#                 best_match_index = np.argmin(face_distances)
#                 if matches[best_match_index]:
#                     name = known_face_names[best_match_index]

#                 face_names.append(name)
#                 # attendance(name)
#                 #currentname = name

#             return name



@app.route('/')
def index():
    global attendance_file_path
    attendance_file_path = data_import.make_folders(config_file["saved_image_folder"],
                                                    config_file["attendance_folder_path"])
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(api_functions.gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/checkin', methods=['POST'])
def checkin():
    name = request.form.get("name")
    status = api_functions.check_in(
        name, attendance_file_path, save_image=True)
    return render_template('result.html', status='Checked In Status For {} : {} '.format(name, status))


@app.route('/checkout', methods=['POST'])
def checkout():
    name = request.form.get("name")
    status = api_functions.check_out(
        name, attendance_file_path, save_image=True)
    return render_template('result.html', status='Checked Out Status {} : {} '.format(name, status))


@app.route('/confirm', methods=['POST'])
def confirm():
    name = api_functions.gen_name()
    return render_template('mid.html', status='You are {} , Press Check In Check Out  '.format(name), name=name)


if __name__ == '__main__':
    app.run(debug=True)

camera.release()
cv2.destroyAllWindows()
