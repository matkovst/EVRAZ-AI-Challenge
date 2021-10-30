import cv2
import numpy as np

def humanConfidenceMask(seg):
    """Формируем маску вероятностей для найденных частей тела.
        
    @param seg обработанный выход модели.

    @return выходная маска размерностью HxW.
    """

    humanConfMask = np.zeros(seg.shape[:2], dtype = np.float32)
    for i in range(1, 7):
        humanConfMask += seg[:, :, i]

    return humanConfMask

def humanMask(seg, thresh = 0.1):
    """Формируем бинарную маску для найденных частей тела.
        
    @param seg обработанный выход модели.
    @param thresh порог для формирования бинарной маски.

    @return выходная маска размерностью HxW.
    """

    segArgmax = np.argmax(seg, axis=-1)
    segMax = np.max(seg, axis=-1)
    segMaxThreshed = (segMax > thresh).astype(np.uint8)
    segArgmax *= segMaxThreshed

    humanMask = np.zeros(segArgmax.shape, dtype = np.uint8)
    for indx in range(1, 7):
        humanMask += np.uint8((segArgmax == indx) * 255)

    return humanMask

def humanBBoxes(seg, thresh = 0.1):
    """Формируем прямоугольники для найденных частей тела.
        
    @param seg обработанный выход модели.
    @param thresh порог для формирования бинарной маски.

    @return выходная маска размерностью HxW.
    """

    hMask = humanMask(seg)
    contours, _ = cv2.findContours(hMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = [None] * len(contours)
    for i, contour in enumerate(contours):
        bboxes[i] = cv2.boundingRect(contour)

    return bboxes

def humanRBGMask(seg, thresh = 0.1):
    """Формируем цветную маску для найденных частей тела.
        
    @param seg обработанный выход модели.
    @param thresh порог для формирования бинарной маски.

    @return выходная маска размерностью HxW.
    """

    segArgmax = np.argmax(seg, axis=-1)
    segMax = np.max(seg, axis=-1)
    segMaxThreshed = (segMax > thresh).astype(np.uint8)
    segArgmax *= segMaxThreshed

    humanMask = np.zeros((segArgmax.shape[0], segArgmax.shape[1], 3), dtype = np.uint8)
    colors = [(0, 0, 255), (0, 127, 255), (0, 255, 255), (0, 255, 0), (255, 0, 0), (211, 0, 148)]
    for indx in range(1, 7):
        humanMask += np.uint8(np.expand_dims(segArgmax == indx, axis = 2) * np.uint8(colors[indx-1]))

    return humanMask

def containsHuman(rbgMask, bbox, thresh = 0.1):
    """Проверяем присутствие человека в прямоугольнике.
        
    @param rbgMask цветная маска частей тела.
    @param thresh порог для формирования бинарной маски.

    @return есть/нету человека.
    """

    roi = rbgMask[bbox[1]:bbox[3] + bbox[1], bbox[0]:bbox[2] + bbox[0]]
    partsNum = np.unique(roi.reshape(-1, roi.shape[-1]), axis=0).shape[0] - 1
    return partsNum > 1

def drawBBoxes(img, bboxes):
    for i in range(len(bboxes)):
        cv2.rectangle(img, bboxes[i], (0, 0, 255), 1)

def drawMask(img, mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (0, 0, 255), -1)