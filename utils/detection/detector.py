from abc import abstractmethod


class Detector:
    def __init__(self):
        pass

    @abstractmethod
    def detect(self,img):
        pass

    def __call__(self, *args, **kwargs):
        return self.detect(*args, **kwargs)