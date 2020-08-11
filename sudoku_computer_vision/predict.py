import cv2
import numpy as np

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = image.reshape(1,28, 28, 3)
    return image


def predict(digits, model):

    output = []
    for image in digits:
        image = preprocess(image)

        if image.sum() > 25000:
            ans = model.predict(image)
            ans = int(np.argmax(ans)) + 1
            output.append(ans)
        else:
            output.append(0)

    return output