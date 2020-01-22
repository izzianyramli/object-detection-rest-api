from PIL import Image
import numpy as np
import settings
import helpers
import flask
import redis
import json
import uuid
import time
import cv2
import io

app = flask.Flask(__name__)
db = redis.StrictRedis(host=settings.REDIS_HOST,
                       port=settings.REDIS_PORT, db=settings.REDIS_DB)

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target)
    image = np.array(image)
    blob = cv2.dnn.blobFromImage(image, 0.007843, (300,300), 127.5)

    return blob

@app.route("/")
def home():
    return ("Hi")


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method == "POST":
        if flask.request.files.get("image"):
            image = flask.request.files["image"].read()

            image = Image.open(io.BytesIO(image))
            image = prepare_image(image, (settings.IMAGE_WIDTH, settings.IMAGE_HEIGHT))
            image = image.copy(order="C")

            k = str(uuid.uuid4())
            d = {"id": k, "image": helpers.base64_encode_image(image)}

            db.rpush(settings.IMAGE_QUEUE, json.dumps(d))

            while True:
                output = db.get(k)

                if output is not None:
                    output = output.decode("utf-8")
                    data["predictions"] = json.loads(output)
                    db.delete(k)
                    break
                time.sleep(settings.CLIENT_SLEEP)
            data["success"] = True

        else:
            flask.jsonify({'error': 'no file'}), 400

    return flask.jsonify(data)


if __name__ == "__main__":
    print("[INFO] Starting web service")
    app.run()
