from flask import Flask, render_template, request, redirect
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2
import recordface

app = Flask(__name__)

vs = VideoStream(src=0).start()
time.sleep(2.0)

outputFrame = None
lock = threading.Lock()


@app.route('/')
def homepage():
    return render_template('index.html')


@app.route('/recordface')
def recface():
    md = recordface()


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=80)
