from flask import Flask, render_template, Response
# from camera import Video
from camera import Video
# from cameraone import Videoone
# from cameratwo import Videotwo

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

def gen(camera):
    while True:
        frame=camera.get_frame()
        yield(b'--frame\r\n'
       b'Content-Type:  image/jpeg\r\n\r\n' + frame +
         b'\r\n\r\n')

@app.route('/video')

def video():
    return Response(gen(Video()),
    mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/videoone')

# def videotwo():
#     return Response(gen(Videoone()),
#     mimetype='multipart/x-mixed-replace; boundary=frame')


# @app.route('/videotwo')

# def videotwo():
#     return Response(gen(Videotwo()),
#     mimetype='multipart/x-mixed-replace; boundary=frame')

app.run(debug=True)
