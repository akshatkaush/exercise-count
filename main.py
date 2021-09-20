from misc.utils import find_person_id_associations
from misc.visualization import draw_points_and_skeleton, joints_dict
from model import SimpleHRNet
import ast
import cv2
import torch
from vidgear.gears import CamGear
import numpy as np
import os
import time
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename
from threading import Thread
import datetime

import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

app = Flask(__name__, static_url_path='')
app.secret_key = os.urandom(42)

OUTPUT_FOLDER = 'downloads'
INPUT_FOLDER = 'uploads'
SITE_URL = 'http://127.0.0.1/'
SENDER_ADDRESS = 'melserbar@gmail.com'
SENDER_PASS = 'some_pass'

session = smtplib.SMTP('smtp.gmail.com', 587)  # use gmail with port
session.starttls()  # enable security
session.login(SENDER_ADDRESS, SENDER_PASS)  # login with mail_id and password


def generate_output(
        input_filename="test.mp4",
        output_filename="output.mp4",
        exercise_type=1,
        email='test@example.com',
        camera_id=0,
        hrnet_weights="./weights/w32_256×192.pth",
        image_resolution="(256,192)",
        hrnet_j=17,
        hrnet_m="HRNet",
        hrnet_c=32,
        hrnet_joints_set="coco",
        single_person=True,
        use_tiny_yolo=False,
        disable_tracking=False,
        max_batch_size=16,
        disable_vidgear=False,
        save_video=True,
        video_format="MJPG",
        video_framerate=30,
        device=None,
):
    if device is not None:
        device = torch.device(device)
    else:
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    image_resolution = ast.literal_eval(image_resolution)
    video_writer = None
    if input_filename is not None:
        video = cv2.VideoCapture(input_filename)
        assert video.isOpened()
    else:
        if disable_vidgear:
            video = cv2.VideoCapture(camera_id)
            assert video.isOpened()
        else:
            video = CamGear(camera_id).start()

    if use_tiny_yolo:
        yolo_model_def = "./models/detectors/yolo/config/yolov3-tiny.cfg"
        yolo_class_path = "./models/detectors/yolo/data/coco.names"
        yolo_weights_path = "./models/detectors/yolo/weights/yolov3-tiny.weights"
    else:
        yolo_model_def = "./models/detectors/yolo/config/yolov3.cfg"
        yolo_class_path = "./models/detectors/yolo/data/coco.names"
        yolo_weights_path = "./models/detectors/yolo/weights/yolov3.weights"

    model = SimpleHRNet(
        hrnet_c,
        hrnet_j,
        hrnet_weights,
        model_name=hrnet_m,
        resolution=image_resolution,
        multiperson=not single_person,
        return_heatmaps=False,
        return_bounding_boxes=not disable_tracking,
        max_batch_size=max_batch_size,
        yolo_model_def=yolo_model_def,
        yolo_class_path=yolo_class_path,
        yolo_weights_path=yolo_weights_path,
        device=device,
    )

    if not disable_tracking:
        prev_boxes = None
        prev_pts = None
        prev_person_ids = None
        next_person_id = 0

    flag = 0
    prev_flag = flag
    counter = 0
    data = 0
    prev_data = data

    while True:
        t = time.time()

        if input_filename is not None or disable_vidgear:
            ret, frame = video.read()
            if not ret:
                break
        else:
            frame = video.read()
            if frame is None:
                break

        pts = model.predict(frame)
        if not disable_tracking:
            boxes, pts = pts
            if len(pts) > 0:
                if prev_pts is None and prev_person_ids is None:
                    person_ids = np.arange(
                        next_person_id, len(pts) + next_person_id, dtype=np.int32
                    )
                    next_person_id = len(pts) + 1
                else:
                    boxes, pts, person_ids = find_person_id_associations(
                        boxes=boxes,
                        pts=pts,
                        prev_boxes=prev_boxes,
                        prev_pts=prev_pts,
                        prev_person_ids=prev_person_ids,
                        next_person_id=next_person_id,
                        pose_alpha=0.2,
                        similarity_threshold=0.4,
                        smoothing_alpha=0.1,
                    )
                    next_person_id = max(next_person_id, np.max(person_ids) + 1)

            else:
                person_ids = np.array((), dtype=np.int32)

            prev_boxes = boxes.copy()
            prev_pts = pts.copy()
            prev_person_ids = person_ids
        else:
            person_ids = np.arange(len(pts), dtype=np.int32)

        for i, (pt, pid) in enumerate(zip(pts, person_ids)):
            frame, data = draw_points_and_skeleton(
                frame,
                pt,
                joints_dict()[hrnet_joints_set]["skeleton"],
                person_index=pid,
                exercise_type=exercise_type,
            )

        frame = cv2.rectangle(
            frame,
            (0, 0),
            (int(frame.shape[1] * 0.7), int(frame.shape[0] * 0.1)),
            (0, 0, 0),
            -1,
        )

        fps = 1.0 / (time.time() - t)
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (int(frame.shape[1] * 0.01), int(frame.shape[0] * 0.035))
        fontScale = frame.shape[0] * 0.0014
        color = (255, 255, 255)
        thickness = 1
        frame = cv2.putText(
            frame,
            "FPS: {:.3f}".format(fps),
            org,
            font,
            fontScale * 0.35,
            color,
            thickness,
            cv2.LINE_AA,
        )

        if exercise_type == 1:  # for pushUps

            if len(pts) > 0:
                if data > 160:
                    flag = 0
                if data < 90:
                    flag = 1
                if prev_flag == 1 and flag == 0:
                    counter = counter + 1

            prev_flag = flag

            org = (int(frame.shape[1] * 0.01), int(frame.shape[0] * 0.08))
            text = "PushUps Count=" + str(counter)
            frame = cv2.putText(
                frame, text, org, font, fontScale, color, thickness * 2, cv2.LINE_AA
            )

        elif exercise_type == 2:  # for Squats

            if len(pts) > 0:
                if data > 150:
                    flag = 0
                if data < 90:
                    flag = 1
                if prev_flag == 1 and flag == 0:
                    counter = counter + 1

            prev_flag = flag

            org = (int(frame.shape[1] * 0.01), int(frame.shape[0] * 0.08))
            text = "Situps Count=" + str(counter)
            frame = cv2.putText(
                frame, text, org, font, fontScale, color, thickness * 2, cv2.LINE_AA
            )

        elif exercise_type == 3:  # for PullUps

            if len(pts) > 0:
                if data == -1 and prev_data == 1:
                    counter = counter + 1

            prev_data = data

            org = (int(frame.shape[1] * 0.01), int(frame.shape[0] * 0.08))
            text = "PullUps Count=" + str(counter)
            frame = cv2.putText(
                frame, text, org, font, fontScale, color, thickness * 2, cv2.LINE_AA
            )

        elif exercise_type == 4:  # for dumbell curl

            if len(pts) > 0:
                if data > 110:
                    flag = 0
                if data < 65:
                    flag = 1
                if prev_flag == 1 and flag == 0:
                    counter = counter + 1

            prev_flag = flag

            org = (int(frame.shape[1] * 0.01), int(frame.shape[0] * 0.08))
            text = "Dumbell Curl Count=" + str(counter)
            frame = cv2.putText(
                frame, text, org, font, fontScale, color, thickness * 2, cv2.LINE_AA
            )

        elif exercise_type == 5:  # for dumbell side lateral

            if len(pts) > 0:
                if data == -1 and prev_data == 1:
                    counter = counter + 1

            prev_data = data

            org = (int(frame.shape[1] * 0.01), int(frame.shape[0] * 0.08))
            text = "Dumbell Side Count=" + str(counter)
            frame = cv2.putText(
                frame, text, org, font, fontScale, color, thickness * 2, cv2.LINE_AA
            )

        if save_video:
            if video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*video_format)  # video format
                video_writer = cv2.VideoWriter(
                    output_filename,
                    fourcc,
                    video_framerate,
                    (frame.shape[1], frame.shape[0]),
                )
            video_writer.write(frame)

    if save_video:
        video_writer.release()
    print("Video processing complete")

    mail_content = f'''Hey,
    Your video has finished processing. You can view your video here : 
    {SITE_URL}{output_filename}
    Thank You
    '''

    message = MIMEMultipart()
    message['From'] = SENDER_ADDRESS
    message['To'] = email
    message['Subject'] = 'Exercise Counter Processing Finished'

    message.attach(MIMEText(mail_content, 'plain'))
    text = message.as_string()
    session.sendmail(SENDER_ADDRESS, email, text)

    print("email sent")


@app.route('/assets/<path:path>')
def send_asset(path):
    return send_from_directory('assets', path)


@app.route(f'/{OUTPUT_FOLDER}/<path:path>')
def download_video(path):
    return send_from_directory(f'{OUTPUT_FOLDER}', path)


@app.route("/", methods=['GET', 'POST'])
def index():
    if request.method == 'GET':

        return render_template('index.html')

    else:
        try:
            uploaded_file = request.files['file']

            exercise_type, email = request.form['exercise_type'], request.form['email']
            filename = secure_filename(
                uploaded_file.filename) + '_' + str(int(datetime.datetime.now().timestamp()))
            uploaded_file.save(INPUT_FOLDER + '/' + filename + '.mp4')
            thread = Thread(
                target=generate_output,
                args=(
                    f"{INPUT_FOLDER}/{filename}.mp4", f"{OUTPUT_FOLDER}/{filename}_output.mp4", int(exercise_type),
                    email)
            )
            thread.daemon = True
            thread.start()

            return render_template('message.html', message='Form submitted successfully.')
        except Exception as _:
            return render_template('message.html', messsage='An error occurred')


if __name__ == "__main__":
    # generate_output(input_filename="uploads/pushup.mp4_1632031449.mp4", output_filename="output.mp4", exercise_type=1)
    app.run("127.0.0.1", 8000)
