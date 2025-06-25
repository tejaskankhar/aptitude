import cv2 
import numpy as np


def resize_by_scale(image, scale_factor):
    
    
    original_height, original_width = image.shape[:2]
    new_dimensions = (
        int(original_width * scale_factor),
        int(original_height * scale_factor)
    )

    resized_image = cv2.resize(image, new_dimensions, interpolation=cv2.INTER_AREA)
    return resized_image



def count_polygon_edges(contour):
    
    arc_len = cv2.arcLength(contour, closed=True)
    epsilon = 0.01 * arc_len
    approx_curve = cv2.approxPolyDP(contour, epsilon, closed=True)
    
    return len(approx_curve)


img1 =cv2.imread('testcase2.py/Bhaskar.jpg')
imgGrey1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2 =cv2.imread('testcase2.py/Ganshyam.jpg')
imgGrey2=cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
img3 =cv2.imread('testcase2.py/Raghav.jpg')
imgGrey3=cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)

grey1=resize_by_scale(imgGrey1,0.5)
grey2=resize_by_scale(imgGrey2,0.5)
grey3=resize_by_scale(imgGrey3,0.5)

stacked= cv2.merge([grey1, grey2, grey3])
normalization = np.uint8(cv2.normalize(stacked, None, 0, 255, cv2.NORM_MINMAX))
merged1 = resize_by_scale(normalization, 0.5)

gray = cv2.cvtColor(merged1, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 1)

bright = cv2.convertScaleAbs(blurred, alpha=1.34, beta=40)

_, thresh = cv2.threshold(bright, 50, 255, cv2.THRESH_BINARY )

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

def get_column(x):
    if x < 100:
        return 'A'
    elif x < 200:
        return 'B'
    elif x < 300:
        return 'C'
    else:
        return 'D'

def get_row(y):
    if y < 100:
        return '1'
    elif y < 160:
        return '2'
    elif y < 220:
        return '3'
    elif y < 260:
        return '4'
    elif y < 300:
        return '5'
    elif y < 360:
        return '6'
    elif y < 400:
        return '7'
    else:
        return '8'

output = bright.copy()

for cnt in contours:
    edges=count_polygon_edges(cnt)
    if edges>10:
     area = cv2.contourArea(cnt)
     if 600 < area < 1000:
        perimeter = cv2.arcLength(cnt, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2 ))
        if circularity > 0.85:
            (x, y), radius = cv2.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)

            cv2.circle(output, center, radius, (255, 225, 255), 2)

            col = get_column(x)
            row = get_row(y)
            label=f"{row}{col}"
            cv2.putText(output,label, (int(x) - 5, int(y) - radius - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 1)
                           
cv2.imshow("Detected Circles", output)
cv2.waitKey(0)
cv2.destroyAllWindows()