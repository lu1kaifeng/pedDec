from flask import Flask
from flask_socketio import SocketIO

from server.Inference import Inference
from server.config import faster_rcnn

i = Inference(config=faster_rcnn)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)
i.run_inference(lambda x:socketio.emit('frame',x))
if __name__ == '__main__':
    socketio.run(app)
