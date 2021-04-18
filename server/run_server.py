from flask import Flask
from flask_socketio import SocketIO

from server.Inference import Inference
from server.config import faster_rcnn

i = Inference(config=faster_rcnn)

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins='*')

@app.before_first_request
def run_infer():
    print('straming...')
    i.run_inference(lambda x: socketio.emit('frame', x),lambda x: socketio.emit('track', x))


if __name__ == '__main__':
    socketio.run(app)
