import os
from flask import Flask, render_template, Response, request
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime
import pandas as pd


# instatiate flask app
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture(0)  # change this if camera not working


path = 'images'
images = []
known_face_names = []
myList = os.listdir(path)
print(myList)
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    known_face_names.append(os.path.splitext(cu_img)[0])
print(known_face_names)


def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList


date = datetime.now().date()
if "Attendance" not in os.listdir(os.getcwd()):
    os.mkdir("Attendance")

exists = os.path.isfile(os.path.join(
    "Attendance", "Attendance_" + str(date) + ".csv"))
if exists:
    print("File There")
else:
    date = datetime.now().date()
    x = open(os.path.join("Attendance", "Attendance_" + str(date) + ".csv"), 'a+')
    data = {'Name': [], 'Time': [], 'Date': [],
            'Check In': [], 'Check Out': [], 'Check Out Time': []}
    df = pd.DataFrame(data, columns=[
                      'Name', 'Time', 'Date', 'Check In', 'Check Out', 'Check Out Time'])  # create DataFrame
    df.set_index('Name', inplace=True)
    df.to_csv(os.path.join("Attendance", "Attendance_" +
              str(date) + ".csv"), sep=',', header=True)


def attendance(name, folder_path=None, save_image=True):
    #ts = time.time()
    date = datetime.now().date()
    #date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
    with open(os.path.join("Attendance", "Attendance_" + str(date) + ".csv"), 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            cin = 1
            f.writelines(f'\n{name},{tStr},{dStr} , {cin}')
            if folder_path and save_image:
                success, frame = camera.read()
                path = f"{folder_path}//{name}"
                if name not in os.listdir(folder_path):
                    os.mkdir(path)
                img_path = f"{path}//{name}_{str(datetime.now().date())}_time-{str(datetime.now().time().strftime('%H-%M-%S'))}_checkin.jpg"
                check = cv2.imwrite(img_path, frame)
                if check:
                    print("Image Saved Successfully")
            checkin_status = "You Successfully Checked In at" + \
                str(time_now.strftime('%H:%M:%S')) + \
                ". Welcome to the Company . Have a Great Day at Work "
            return checkin_status
        else:
            df = pd.read_csv("Attendance_" + str(date) +
                             ".csv", index_col='Name')
            if(df['Check Out'][name] == 1):
                checkin_status = "You Already Checked Out at" + \
                    str(df['Check Out Time'][name]) + "! See You Tomorrow :) "
                #checkin_status = "You Already Checked Out at" + str(df['Check Out Time'][name]) + "! See You Tomorrow :) "
            elif(df['Check In'][name] == 1):
                checkin_status = "You Already Checked in at" + \
                    str(df['Time'][name]) + \
                    "! You can now Checked Out Only :) "
                #checkin_status = "You Already Checked in at" + str(df['Time'][name]) + "! You can now Checked Out Only :) "
            return checkin_status


def fcheckout(name, folder_path=None, save_image=True):
    checkout_status = "OK"
    date = datetime.now().date()
    df = pd.read_csv("Attendance_" + str(date) + ".csv", index_col='Name')
    df.head()
    with open(os.path.join("Attendance", "Attendance_" + str(date) + ".csv"), 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            checkout_status = "You Have not Checked In Yet"

        else:
            if(df['Check Out'][name] == 1):
                checkout_status = "You Already Checked Out at " + \
                    str(df['Check Out Time'][name])
            else:
                df['Check Out'][name] = 1
                time_now = datetime.now()
                df['Check Out Time'][name] = time_now.strftime('%H:%M:%S')
                if folder_path and save_image:
                    success, frame = camera.read()
                    path = f"{folder_path}//{name}"
                    if name not in os.listdir(folder_path):
                        os.mkdir(path)
                    img_path = f"{path}//{name}_{str(datetime.now().date())}_time-{str(datetime.now().time().strftime('%H-%M-%S'))}_checkout.jpg"
                    check = cv2.imwrite(img_path, frame)
                    if check:
                        print("Image Saved Successfully")
                checkout_status = "Successfully Checked Out at " + \
                    str(df['Check Out Time'][name])

    df.to_csv("Attendance_" + str(date) + ".csv", sep=',', header=True)

    return checkout_status


known_face_encodings = faceEncodings(images)


# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True
currentname = "U"


def gen_frames():
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time

            # Find all the faces and face encodings in the current frame of video
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding)
                name = "Unknown"
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
                # attendance(name)
                currentname = name

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


def gen_name():
    while True:
        success, frame = camera.read()  # read the camera frame
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
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(
                rgb_small_frame, face_locations)
            face_names = []
            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(
                    known_face_encodings, face_encoding)
                # Or instead, use the known face with the smallest distance to the new face
                face_distances = face_recognition.face_distance(
                    known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)
                # attendance(name)
                #currentname = name

            return name


gn = "None"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/checkin', methods=['POST'])
def checkin():
    if "saved_images" not in os.listdir():
        os.mkdir(os.path.join(os.getcwd(), "saved_images"))

    #n = gen_name()
    n = request.form.get("take")
    stat = attendance(n, folder_path="saved_images", save_image=True)
    return render_template('result.html', status='Checked In Status For {} : {} '.format(n, stat))


@app.route('/checkout', methods=['POST'])
def checkout():
    if "saved_images" not in os.listdir():
        os.mkdir(os.path.join(os.getcwd(), "saved_images"))
        #n = gen_name()
    n = request.form.get("take")
    stat = fcheckout(n, folder_path="saved_images", save_image=True)
    return render_template('result.html', status='Checked Out Status {} : {} '.format(n, stat))


@app.route('/confirm', methods=['POST'])
def confirm():
    gn = gen_name()
    return render_template('mid.html', status='You are {} , Press Check In Check Out  '.format(gn), n=gn)


if __name__ == '__main__':
    app.run()

camera.release()
cv2.destroyAllWindows()
