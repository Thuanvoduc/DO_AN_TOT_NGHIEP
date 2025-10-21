import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import urllib.request
import numpy as np

url = 'http://192.168.1.5/cam-hi.jpg'

cv2.namedWindow("live transmission", cv2.WINDOW_AUTOSIZE)

while True:
    # Lấy hình từ ESP32-CAM
    img_resp = urllib.request.urlopen(url)
    imgnp = np.array(bytearray(img_resp.read()), dtype=np.uint8)
    frame = cv2.imdecode(imgnp, -1)

    # Face detection
    faces, confidences = cv.detect_face(frame)
    # Vẽ bounding box lên frame
    for face, confidence in zip(faces, confidences):
        x1, y1, x2, y2 = face
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'{int(confidence*100)}%', (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    # Hiển thị live feed có face detection
    cv2.imshow("live transmission", frame)

    # Nhấn 'q' để thoát
    key = cv2.waitKey(5)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
