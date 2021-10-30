import cv2

def drawBBoxes(img, bboxes):
    for i in range(len(bboxes)):
        cv2.rectangle(img, bboxes[i], (0, 0, 255), 1)

def drawMask(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), -1)