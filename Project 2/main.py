import flask
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True



from darknet import *
import cv2
import numpy as np
import PIL
import io
import html
import time
import matplotlib.pyplot as plt

from flask import Flask, render_template, session, copy_current_request_context
from flask_socketio import SocketIO, emit, disconnect
from threading import Lock


async_mode = None
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socket_ = SocketIO(app, async_mode=async_mode)
thread = None
thread_lock = Lock()


@app.route('/')
def index():
    return render_template('index.html', async_mode=socket_.async_mode)


@socket_.on('my_event', namespace='/test')
def test_message(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']})


@socket_.on('my_broadcast_event', namespace='/test')
def test_broadcast_message(message):
    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': message['data'], 'count': session['receive_count']},
         broadcast=True)


@socket_.on('disconnect_request', namespace='/test')
def disconnect_request():
    @copy_current_request_context
    def can_disconnect():
        disconnect()

    session['receive_count'] = session.get('receive_count', 0) + 1
    emit('my_response',
         {'data': 'Disconnected!', 'count': session['receive_count']},
         callback=can_disconnect)




#@app.route('/', methods=['GET'])
#def home():
  #  return '''<h1>Distant Reading Archive</h1>
#<p>A prototype API for distant reading of science fiction novels.</p>'''


# import darknet functions to perform object detections

# load in our YOLOv4 architecture network
network, class_names, class_colors = load_network("cfg/yolov4-csp.cfg", "cfg/coco.data", "yolov4-csp.weights")
width = network_width(network)
height = network_height(network)

# darknet helper function to run detection on image
def darknet_helper(img, width, height):
  darknet_image = make_image(width, height, 3)
  img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  img_resized = cv2.resize(img_rgb, (width, height),
                              interpolation=cv2.INTER_LINEAR)

  # get image ratios to convert bounding boxes to proper size
  img_height, img_width, _ = img.shape
  width_ratio = img_width/width
  height_ratio = img_height/height

  # run model on darknet style image to get detections
  copy_image_from_bytes(darknet_image, img_resized.tobytes())
  detections = detect_image(network, class_names, darknet_image)
  free_image(darknet_image)
  return detections, width_ratio, height_ratio



def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)


# Opens the inbuilt camera of laptop to capture video.
cap = cv2.VideoCapture(0)
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
# initialze bounding box to empty
#out = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (frame_width, frame_height))
bbox = ''
count = 0

if __name__ == '__main__':
    socket_.run(app, debug=True)
while True:
    pre_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    if ret:
        cv2.imwrite('frame{:d}.jpg'.format(count), frame)
        count += 30  # i.e. at 30 fps, this advances one second
        cap.set(1, count)
    detections, width_ratio, height_ratio = darknet_helper(frame, width, height)

    # loop through detections and draw them on transparent overlay image
    for label, confidence, bbox in detections:
        left, top, right, bottom = bbox2points(bbox)
        left, top, right, bottom = int(left * width_ratio), int(top * height_ratio), int(right * width_ratio), int(
            bottom * height_ratio)
        cv2.rectangle(frame, (left, top), (right, bottom), class_colors[label], 2)
        cv2.putText(frame, "{} [{:.2f}]".format(label, float(confidence)),
                    (left, top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    class_colors[label], 2)

        message = label
    cv2.imshow('Demo', frame)
    cv2.waitKey(1)
    #out.write(frame)



app.run()


cap.release()




if __name__ == '__main__':
    socket_.run(app, debug=True)