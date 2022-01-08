from flask import Flask, render_template, Response, request, session, redirect, url_for
import os
import cv2
import json
import bcrypt
import pymongo
from datetime import timedelta

# custom package imports
from data_ingestion.data_import_and_preprocessing import DataImport, Preprocessing
# from attendance_api_functions.api_functions import API_Functions
from database_api_functions.db_api_functions import DatabaseAPI

#dashboard app
from dashboard.dashboard import dashboard_app

# instatiate flask app
app = Flask(__name__, template_folder="templates")
#registering the dashboard app
app.register_blueprint(dashboard_app)
#secret key
app.secret_key = "secret!"

#for setting the session time
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(minutes=30)

# making objects
camera = cv2.VideoCapture(0)  # change this if camera not working
data_import = DataImport()
preprocessing = Preprocessing()

# loading config.json
with open("config.json") as f:
    config_file = json.load(f)

# making all the folders
data_import.make_folders(config_file["saved_image_folder"],
                         config_file["attendance_folder_path"],
                         config_file["image_path"])

# calling important functions
images, known_face_names = data_import.read_images(
    image_path=config_file["image_path"])
known_face_encodings = preprocessing.faceEncodings(images)

# creating database object
database = DatabaseAPI(camera, known_face_names, known_face_encodings,
                       config_file["mongo_db_connection_url"],
                       config_file["attendance_database_name"],
                       config_file["saved_image_folder"])

# creating collection
database.make_database_collection()

#mongodb connection
mongodb_url = config_file["mongo_db_connection_url"]
client = pymongo.MongoClient(mongodb_url)

# api_functions = API_Functions(camera, known_face_names, known_face_encodings,
#                               config_file["saved_image_folder"])

#login database
login_collection_name = config_file[
    "login_collection_name"]  #collection name from config
login_database = client[config_file["login_database_name"]]
login_collection = login_database[login_collection_name]


############################# Login #############################
@app.route('/')
def index():
    global login_collection  #using the global scope variables
    login_collection = login_database[
        login_collection_name]  #fetching data from database
    if 'username' in session:
        return redirect(url_for('home'))

    return render_template('login/login.html', alert=False)


@app.route('/login', methods=['POST'])
def login():
    if request.method == "POST":
        if request.form:
            login_user = login_collection.find_one(
                {'Email':
                 request.form['email'].lower()})  #find user in database
            if login_user:
                if bcrypt.checkpw(request.form['password'].encode('utf8'),
                                  login_user['Password']):
                    session['username'] = login_user['Name']
                    session['email'] = login_user['Email']
                    session.permanent = True
                    return redirect(url_for('index'))

            return render_template(
                'login/login.html',
                alert=True,
                alert_msg="Invalid username/password combination")
    else:
        return "Invalid Request"


@app.route('/change_password', methods=['POST'])
def change_password():
    if request.method == 'POST':
        if request.form:
            global login_collection
            user_details = login_collection.find_one(
                {'Email': session["email"]})
            if request.form["new_password"] == request.form[
                    "confirm_new_password"]:
                if bcrypt.checkpw(
                        request.form['current_password'].encode('utf8'),
                        user_details['Password']):
                    new_password = bcrypt.hashpw(
                        request.form['new_password'].encode('utf-8'),
                        bcrypt.gensalt())
                    login_collection.update_one(
                        user_details, {"$set": {
                            "Password": new_password
                        }})
                    return redirect(url_for('logout'))
                else:
                    return "Current password does not match!!"

            else:
                return "New Password and Confirm New Password does not match"

    return redirect(url_for('index'))


@app.route('/logout')
def logout():
    if 'username' in session:
        try:
            session.clear()  #clearing all the session variables
            return redirect(url_for('index'))
        except KeyError:
            return redirect(url_for('index'))
    else:
        return redirect(url_for('index'))


###################################################################
@app.route("/home")
def home():
    if "username" in session:
        database.make_database_collection()
        return render_template("attendance-templates//index.html")

    return redirect(url_for('index'))


@app.route("/video_feed")
def video_feed():
    return Response(database.gen_frames(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/checkin", methods=["POST"])
def checkin():
    if "username" in session:
        name = session["username"]
        status = database.check_in(name, save_image=True)
        return render_template("attendance-templates//result.html",
                               status="Checked In Status For {} : {} ".format(
                                   name, status))

    return redirect(url_for('index'))


@app.route("/checkout", methods=["POST"])
def checkout():
    if "username" in session:
        name = session["username"]
        status = database.check_out(name, save_image=True)
        return render_template("attendance-templates//result.html",
                               status="Checked Out Status {} : {} ".format(
                                   name, status))

    return redirect(url_for('index'))


@app.route("/confirm", methods=["POST"])
def confirm():
    if "username" in session:
        name = database.gen_name()
        if name != session["username"]:
            return render_template(
                "attendance-templates//unknown.html",
                status="You are not " + session["username"] +
                "!! If you think this is a mistake contact the administrator")
        return render_template(
            "attendance-templates//mid.html",
            status="You are {} , Press Check In Check Out  ".format(name),
            name=name)

    return redirect(url_for('index'))


@app.route('/update')
def update():
    if "username" in session:
        try:
            return render_template('attendance-templates//update.html',
                                   alert=False)
        except Exception as e:
            print(e)
    return redirect(url_for('index'))


@app.route('/upload', methods=['POST'])
def upload():
    if "username" in session:
        try:
            if request.files and "image" in request.files["image"].content_type:
                image = request.files["image"]
                img_ext = os.path.splitext(image.filename)[1]
                #deleting previous image
                images_list = os.listdir(config_file["image_path"])
                for img in images_list:
                    if session["username"] == os.path.splitext(img)[0]:
                        os.remove(os.path.join(config_file["image_path"], img))
                #saving new image
                path = os.path.join(config_file["image_path"],
                                    session["username"] + img_ext)
                image.save(path)
                global database, images, known_face_names, known_face_encodings
                
                images,known_face_names = data_import.read_images()
                known_face_encodings = preprocessing.faceEncodings(images)
                database = DatabaseAPI(camera, known_face_names,
                                       known_face_encodings,
                                       config_file["mongo_db_connection_url"],
                                       config_file["attendance_database_name"],
                                       config_file["saved_image_folder"])
                return render_template("attendance-templates//update.html",
                                       alert=True)
            else:
                return render_template("attendance-templates//update.html",wrong_type=True)
            
        except Exception as e:
            print(e)
            raise e

    return redirect(url_for('index'))


if __name__ == "__main__":
    try:
        app.run(debug=True)

    except Exception:
        print("Stopping the Program!!!!")

    finally:
        camera.release()
        cv2.destroyAllWindows()