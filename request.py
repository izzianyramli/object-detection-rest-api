from datetime import datetime
import numpy as np
import argparse
import requests
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                help="Required. path to input image.", required=True, type=str)
args = vars(ap.parse_args())


URL = "http://localhost:5000/predict"
IMAGE_PATH = args["image"]

COLORS = np.random.uniform(0, 255, size=(21, 3))


print("[INFO] Image file name: {}".format(IMAGE_PATH))
image = open(IMAGE_PATH, "rb").read()
payload = {"image": image}

image = cv2.imread(IMAGE_PATH)
(h,w) = image.shape[:2]


print("[INFO] Loading image to server")
start_time = datetime.now()
r = requests.post(URL, files=payload).json()
print("[INFO] Image passed to server")

if r["success"]:
    for (i, result) in enumerate(r["predictions"]):
        print("{}. {}: {:.4f}".format(
            i+1, result["label"], result["probability"]))

        startX = int(np.array(result["startX"]*w))
        startY = int(np.array(result["startY"]*h))
        endX = int(np.array(result["endX"]*w))
        endY = int(np.array(result["endY"]*h))

        label = "{}: {:.2f}%".format(result["label"], result["probability"]*100)

        cv2.rectangle(image, (startX, startY), (endX, endY),
                      color=[255, 0, 0], thickness=2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(image, label, (startX, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color=[255, 0, 0], thickness=2)
    cv2.imshow("output", image)
    cv2.waitKey(0)


else:
    print("Request failed")

total_time = "{:.2f}".format((datetime.now() - start_time).total_seconds())
print("[INFO] Time interval: {} seconds".format(total_time))
