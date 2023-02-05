import json
import cv2
import numpy as np

img = cv2.imread('lib/png_image.png')
with open("lib/camera_intrinsics.json", "r") as f:
    data = json.load(f)
fx = data["ffx"]
real_world_distance = 380

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, threshold = cv2.threshold(img_gray, 50, 255, cv2.CHAIN_APPROX_NONE)
contours, _ = cv2.findContours(threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


def calc_triangle():
    vec1 = np.array([approx[0][0][0] - approx[1][0][0], approx[0][0][1] - approx[1][0][1]])
    vec2 = np.array([approx[2][0][0] - approx[1][0][0], approx[2][0][1] - approx[1][0][1]])
    cosine_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    angle = np.degrees(np.arccos(cosine_angle))
    angle = int(np.around(angle))
    cv2.putText(img, "Triangle angle is " + str(angle), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))


def calc_four_angle():
    x, y, w, h = cv2.boundingRect(approx)
    aspect_ratio = float(w) / h
    side_length = max(w, h)
    size_in_mm = (side_length * real_world_distance) / fx
    size_in_mm = int(np.round(size_in_mm))
    if 0.95 <= aspect_ratio < 1.05:
        cv2.putText(img, "square side length is " + str(size_in_mm) + "mm", (x, y),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))
    else:
        cv2.putText(img, "rectangle side length is " + str(side_length) + "mm", (x, y),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0))


def calc_circle():
    (x, y), radius = cv2.minEnclosingCircle(approx)
    radius_mm = (radius * real_world_distance) / fx
    radius_mm = int(np.round(radius_mm))
    cv2.putText(img, "circle radius is " + str(radius_mm) + "mm",
                (int(x), int(y + radius_mm * 3)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)


for contour in contours:
    approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
    cv2.drawContours(img, [approx], 0, (0, 255, 255), 2)
    x = approx.ravel()[0]
    y = approx.ravel()[1] - 5
    if len(approx) == 3:
        calc_triangle()
    elif len(approx) == 4:
        calc_four_angle()
    else:
        calc_circle()

cv2.putText(img, "Total shapes found: " + str(len(contours)), (10, 700), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
cv2.imshow('detected_shapes', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
