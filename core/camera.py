import cv2
import time
from threading import Thread

class CameraStream:
    def __init__(self, src, name="Camera"):
        self.stream = cv2.VideoCapture(src)
        self.ret, self.frame = self.stream.read()
        self.stopped = False
        self.name = name
        
        # Start the thread
        self.t = Thread(target=self.update, args=())
        self.t.daemon = True
        self.t.start()

    def update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if ret:
                self.frame = frame
                self.ret = True
            else:
                self.ret = False
                # Optional: Add your reconnect logic here

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.t.join()
        self.stream.release()