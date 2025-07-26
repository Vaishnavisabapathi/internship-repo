import cv2
import numpy as np

def projection_profile(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    horizontal_projection = np.sum(bin_img, axis=1)
    lines = []
    in_line = False
    start = 0

    for i, row_sum in enumerate(horizontal_projection):
        if row_sum > 0 and not in_line:
            start = i
            in_line = True
        elif row_sum == 0 and in_line:
            end = i
            in_line = False
            line_img = image[start:end, :]
            if line_img.shape[0] > 5:  # filter out noise
                lines.append(line_img)

    return lines
