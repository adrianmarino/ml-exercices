import cv2


class Video:
    def __init__(self, path): self.cap = cv2.VideoCapture(path)

    def images(self, process):
        images = []
        ret, image = self.cap.read()
        while ret:
            images.append(process(image))
            ret, image = self.cap.read()
        return images

    def close(self): self.cap.release()
