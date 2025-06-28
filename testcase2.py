import cv2 
import numpy as np


def resize_by_scale(image, scale_factor):
    
    
    original_height, original_width = image.shape[:2]
    new_dimensions = (
        int(original_width * scale_factor),
        int(original_height * scale_factor)
    )
def is_filled_white(contour, gray_img, min_mean=200):
    mask = np.zeros_like(gray_img, dtype=np.uint8)
    
    cv2.drawContours(mask, [contour], contourIdx=-1, color=255, thickness=cv2.FILLED)
    
    pixel_values = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    selected_pixels = pixel_values[mask == 255]
    
    if selected_pixels.size == 0:
        return False
    
    return np.mean(selected_pixels) > min_mean
def get_column(x):
    if x < 150:
        return 'A'
    elif x < 250:
        return 'B'
    elif x < 400:
        return 'C'
    else:
        return 'D'

def get_row(y):
    if y < 100:
        return '1'
    elif y < 150:
        return '2'
    elif y < 190:
        return '3'
    elif y < 230:
        return '4'
    elif y < 270:
        return '5'
    elif y < 300:
        return '6'
    elif y < 350:
        return '7'
    elif y < 380:
        return '8'
    elif y < 400:
        return '9'
    else:
        return '10'

imgGrey1 = cv2.imread('Bhaskar (1).jpg')
imgGrey2 = cv2.imread('Ganshyam (1).jpg')
imgGrey3= cv2.imread('Raghav (1).jpg')

gray1 = cv2.cvtColor(imgGrey1, cv2.COLOR_BGR2GRAY)
gray1 = cv2.resize(gray1, (500, 500))
gray2 = cv2.cvtColor(imgGrey2, cv2.COLOR_BGR2GRAY)
gray2 = cv2.resize(gray2, (500, 500))
gray3 = cv2.cvtColor(imgGrey3, cv2.COLOR_BGR2GRAY)
gray3 = cv2.resize(gray3, (500, 500))

bright1 = cv2.convertScaleAbs(gray1, alpha=2.5, beta=50)
bright3 = cv2.convertScaleAbs(gray3, alpha=1, beta=50)
stacked = cv2.merge([bright1, gray2, bright3])
final_img = cv2.cvtColor(stacked, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(final_img, (5, 5), 1)
_, thresh = cv2.threshold(blurred, 28, 255, cv2.THRESH_BINARY)
cv2.imshow("Threshold", thresh)

contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
output = cv2.cvtColor(final_img.copy(), cv2.COLOR_GRAY2BGR)

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    
    if x <= 50 or y <= 50:
        continue 
    area = cv2.contourArea(contour)
    
    if area > 300 and is_filled_white(contour, thresh):
        cx, cy = x + w // 2, y + h // 2
        label = f"{get_row(cy)}{get_column(cx)}"
        print(label)

        cv2.drawContours(thresh, [contour], -1, (0, 255, 0), 2)
        cv2.putText(output, label, (x, y - 10),
                    cv2.FONT_ITALIC, 0.5, (0, 255, 0), 1)

cv2.imshow("Detected Filled Shapes", output)

cv2.waitKey(0)
cv2.destroyAllWindows()
