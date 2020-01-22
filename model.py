import numpy as np
import argparse
import flask
import redis
import json
import helpers
import time
import settings
import cv2

app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)


def classify_process():
    print("[INFO] Loading model")
    net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
    print("[INFO] Model loaded")

    while True:
        queue = db.lrange(settings.IMAGE_QUEUE, 0, settings.BATCH_SIZE - 1)
        imageIDs = []
        batch = None

        for q in queue:
            q = json.loads(q.decode("utf-8"))
            image = helpers.base64_decode_image(q["image"], settings.IMAGE_DTYPE, (
                1, settings.IMAGE_CHANS, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH))

            if batch is None:
                batch = image
            else:
                batch = np.vstack([batch, image])

            imageIDs.append(q["id"])

        if len(imageIDs) > 0:
            print("[INFO] Batch size: {}".format(batch.shape))
            print("[INFO] Computing object detection")
            net.setInput(batch)
            detections = net.forward()

            output = []

            for imageID in imageIDs:
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]

                    if confidence > args["confidence"]:
                        idx = int(detections[0, 0, i, 1])
                        box = detections[0, 0, i, 3:7]
                        (startX, startY, endX, endY) = box

                        r = {"label": CLASSES[idx],
                             "probability": float(confidence),
                             "startX":float(startX), "startY": float(startY),
                             "endX": float(endX), "endY": float(endY)}
                        output.append(r)

                db.set(imageID, json.dumps(output))
            db.ltrim(settings.IMAGE_QUEUE, len(imageIDs), -1)

        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--prototxt", required=True,
                    help="path to Caffe 'deploy' prototxt file")
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-c", "--confidence", type=float, default=0.2,
                    help="minimum probability to filter weak detections")
    args = vars(ap.parse_args())

    # initialize list of class labels MobileNet SSD
    CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
               "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

    classify_process()
