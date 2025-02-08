import dlib
import numpy as np
import cv2
import os
import pandas as pd
import time
import logging
import mysql.connector
import datetime


# Dlib / Use frontal face detector of Dlib
detector = dlib.get_frontal_face_detector()

# Dlib landmark / Get face landmarks
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# Dlib Resnet Use Dlib resnet50 model to get 128D face descriptor
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


# MySQL connection
def get_db_connection():
    try:
        conn = mysql.connector.connect(
            host='51.20.6.164',
            user='krish',
            password='krish@ml', 
            database='attendance' 
        )
        return conn
    except mysql.connector.Error as err:
        logging.error(f"Error connecting to MySQL: {err}")
        return None


# Create a table for the current date
def create_table():
    conn = get_db_connection()
    if conn is None:
        return
    cursor = conn.cursor()
    current_date = datetime.datetime.now().strftime("%Y_%m_%d")
    table_name = "attendance"
    create_table_sql = f"""
    CREATE TABLE IF NOT EXISTS {table_name} (
        name VARCHAR(255),
        time TIME,
        date DATE,
        UNIQUE(name, date)
    )
    """
    try:
        cursor.execute(create_table_sql)
        conn.commit()
    except mysql.connector.Error as err:
        logging.error(f"Error creating table: {err}")
    finally:
        cursor.close()
        conn.close()


class Face_Recognizer:
    def __init__(self):
        self.font = cv2.FONT_ITALIC

        # FPS
        self.frame_time = 0
        self.frame_start_time = 0
        self.fps = 0
        self.fps_show = 0
        self.start_time = time.time()

        # cnt for frame
        self.frame_cnt = 0

        # Save the features of faces in the database
        self.face_features_known_list = []
        # Save the name of faces in the database
        self.face_name_known_list = []

        # List to save centroid positions of ROI in frame N-1 and N
        self.last_frame_face_centroid_list = []
        self.current_frame_face_centroid_list = []

        # List to save names of objects in frame N-1 and N
        self.last_frame_face_name_list = []
        self.current_frame_face_name_list = []

        # cnt for faces in frame N-1 and N
        self.last_frame_face_cnt = 0
        self.current_frame_face_cnt = 0

        # Save the e-distance for faceX when recognizing
        self.current_frame_face_X_e_distance_list = []

        # Save the positions and names of current faces captured
        self.current_frame_face_position_list = []
        # Save the features of people in current frame
        self.current_frame_face_feature_list = []

        # e distance between centroid of ROI in last and current frame
        self.last_current_frame_centroid_e_distance = 0

        # Reclassify after 'reclassify_interval' frames
        self.reclassify_interval_cnt = 0
        self.reclassify_interval = 10

        # Create table for attendance if it does not exist
        create_table()

    # Get known faces from "features_all.csv"
    def get_face_database(self):
        if os.path.exists("data/features_all.csv"):
            path_features_known_csv = "data/features_all.csv"
            csv_rd = pd.read_csv(path_features_known_csv, header=None)
            for i in range(csv_rd.shape[0]):
                features_someone_arr = []
                self.face_name_known_list.append(csv_rd.iloc[i][0])
                for j in range(1, 129):
                    if csv_rd.iloc[i][j] == '':
                        features_someone_arr.append('0')
                    else:
                        features_someone_arr.append(csv_rd.iloc[i][j])
                self.face_features_known_list.append(features_someone_arr)
            logging.info("Faces in Database： %d", len(self.face_features_known_list))
            return 1
        else:
            logging.warning("'features_all.csv' not found!")
            logging.warning("Please run 'get_faces_from_camera.py' "
                            "and 'features_extraction_to_csv.py' before 'face_reco_from_camera.py'")
            return 0

    def update_fps(self):
        now = time.time()
        # Refresh fps per second
        if str(self.start_time).split(".")[0] != str(now).split(".")[0]:
            self.fps_show = self.fps
        self.start_time = now
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    @staticmethod
    # Compute the e-distance between two 128D features
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # Use centroid tracker to link face_x in current frame with person_x in last frame
    def centroid_tracker(self):
        for i in range(len(self.current_frame_face_centroid_list)):
            e_distance_current_frame_person_x_list = []
            # For object 1 in current_frame, compute e-distance with object 1/2/3/4/... in last frame
            for j in range(len(self.last_frame_face_centroid_list)):
                self.last_current_frame_centroid_e_distance = self.return_euclidean_distance(
                    self.current_frame_face_centroid_list[i], self.last_frame_face_centroid_list[j])

                e_distance_current_frame_person_x_list.append(
                    self.last_current_frame_centroid_e_distance)

            last_frame_num = e_distance_current_frame_person_x_list.index(
                min(e_distance_current_frame_person_x_list))
            self.current_frame_face_name_list[i] = self.last_frame_face_name_list[last_frame_num]

    # cv2 window / putText on cv2 window
    def draw_note(self, img_rd):
        # Add some info on windows
        cv2.putText(img_rd, "Face Recognizer with Deep Learning", (20, 40), self.font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Frame:  " + str(self.frame_cnt), (20, 100), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "FPS:    " + str(self.fps.__round__(2)), (20, 130), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces:  " + str(self.current_frame_face_cnt), (20, 160), self.font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), self.font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

        for i in range(len(self.current_frame_face_name_list)):
            img_rd = cv2.putText(img_rd, "Face_" + str(i + 1), tuple(
                [int(self.current_frame_face_centroid_list[i][0]), int(self.current_frame_face_centroid_list[i][1])]),
                                 self.font,
                                 0.8, (255, 190, 0),
                                 1,
                                 cv2.LINE_AA)

    # Insert data in database
    def attendance(self, name):
        current_date = datetime.datetime.now().strftime('%Y-%m-%d')
        conn = get_db_connection()
        if conn is None:
            return
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT * FROM attendance WHERE name = %s AND date = %s", (name, current_date))
            existing_entry = cursor.fetchone()

            if existing_entry:
                print(f"{name} is already marked as present for {current_date}")
            else:
                current_time = datetime.datetime.now().strftime('%H:%M:%S')
                cursor.execute("INSERT INTO attendance (name, time, date) VALUES (%s, %s, %s)", (name, current_time, current_date))
                conn.commit()
                print(f"{name} marked as present for {current_date} at {current_time}")
        except mysql.connector.Error as err:
            logging.error(f"Error inserting attendance record: {err}")
        finally:
            cursor.close()
            conn.close()

    # Face detection and recognition with OT from input video stream
    def process(self, stream):
        # Get faces known from "features_all.csv"
        if self.get_face_database():
            while stream.isOpened():
                self.frame_cnt += 1
                logging.debug("Frame " + str(self.frame_cnt) + " starts")
                flag, img_rd = stream.read()
                kk = cv2.waitKey(1)

                # Detect faces for frame X
                faces = detector(img_rd, 0)

                # Update cnt for faces in frames
                self.current_frame_face_cnt = len(faces)

                self.current_frame_face_name_list = ["unknown" for _ in range(len(faces))]
                self.current_frame_face_centroid_list = []
                self.current_frame_face_feature_list = []

                for i, face in enumerate(faces):
                    shape = predictor(img_rd, face)
                    face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
                    face_descriptor = np.array(face_descriptor)

                    # Compute 128D feature of face
                    feature_current_frame_face = face_descriptor.tolist()
                    self.current_frame_face_feature_list.append(feature_current_frame_face)
                    self.current_frame_face_centroid_list.append(
                        [int((face.left() + face.right()) / 2), int((face.top() + face.bottom()) / 2)])

                if self.frame_cnt % self.reclassify_interval == 0:
                    logging.debug("Reclassify...")
                    for i in range(len(self.current_frame_face_feature_list)):
                        e_distance_list = []
                        for j in range(len(self.face_features_known_list)):
                            e_distance_list.append(self.return_euclidean_distance(
                                self.current_frame_face_feature_list[i], self.face_features_known_list[j]))
                        self.current_frame_face_name_list[i] = self.face_name_known_list[e_distance_list.index(min(e_distance_list))]

                # Update current frame to last frame
                self.last_frame_face_centroid_list = self.current_frame_face_centroid_list
                self.last_frame_face_name_list = self.current_frame_face_name_list
                self.last_frame_face_cnt = self.current_frame_face_cnt

                # Use centroid tracker
                self.centroid_tracker()

                # Attendance Check
                for i in range(len(self.current_frame_face_name_list)):
                    if self.current_frame_face_name_list[i] != 'unknown':
                        self.attendance(self.current_frame_face_name_list[i])
                self.draw_note(img_rd)
                cv2.imshow("Face Recognizer", img_rd)
                if kk == ord('q'):
                    break

                # Update fps
                self.update_fps()

        # Release video capture
        stream.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    face_recognizer = Face_Recognizer()
    video_stream = cv2.VideoCapture(0)
    face_recognizer.process(video_stream)
