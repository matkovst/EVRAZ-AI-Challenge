# -*- coding: utf-8 -*-

import argparse
import sys
import os
import cv2
import json
import numpy as np
from CDCL.model_specific import get_testing_model_resnet101
import utils

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


class CDCL7:
    """Класс для работы с моделью CDCL7 (https://github.com/kevinlin311tw/CDCL-human-part-segmentation).
    Позволяет получить сегментные маски семи частей тела человека и заключить всего человека в прямоугольник.
    """
    
    def __init__(self, args):
        """Конструктор.
        
        @param args словарь с входными параметрами.
        """

        # Подготавливаем параметры
        self.stride = 8 # значение по-умолчанию из оригинальной модели

        self.multiplier = []
        for item in set(args.scale):
            self.multiplier.append(float(item))

        self.partsNum = 7 # 6 частей тела + background
        self.humanPart = [0, 1, 2, 3, 4, 5, 6]

        self.verbose = bool(args.verbose)

        # Подготавливаем модель
        print('>>> Initializing model ...')
        self.model = None
        try:
            self.model = get_testing_model_resnet101()
            weights = args.model
            self.model.load_weights(weights)
        except Exception as e:
            print('>>> Could not initialize model:', e)
        else:
            print('>>> Model initialized successfully')

    def Infer(self, rawImg):
        """Запускаем модель на входном изображении.
        
        @param img входное изображение в формате RGB.

        @return изображение сегментной маски размерностью HxWx7.
        """
        
        if self.model is None:
            return None

        if rawImg is None:
            return None
        
        # Предобрабатываем изображение для модели
        H = rawImg.shape[0]
        W = rawImg.shape[1]
        img_f32 = (rawImg / 256.0) - 0.5
        flippedImg_f32 = cv2.flip(rawImg, 1)
        flippedImg_f32 = (flippedImg_f32 / 256.0) - 0.5

        segmapScale1 = np.zeros((H, W, self.partsNum))
        segmapScale2 = np.zeros((H, W, self.partsNum))
        segmapScale3 = np.zeros((H, W, self.partsNum))
        segmapScale4 = np.zeros((H, W, self.partsNum))
        segmapScale5 = np.zeros((H, W, self.partsNum))
        segmapScale6 = np.zeros((H, W, self.partsNum))
        segmapScale7 = np.zeros((H, W, self.partsNum))
        segmapScale8 = np.zeros((H, W, self.partsNum))

        # Запускаем модель на оригинальной ориентации
        for m, scale in enumerate(self.multiplier):

            # Формируем входной блоб
            inputBlob = self.__makeInputBlob(img_f32, scale)

            # Колдуем
            outputBlobs = self.model.predict(inputBlob)
            if self.verbose:
                print( "\tProcessed blob with size:", inputBlob.shape)

            # Делаем пост-обработку результата
            seg = self.__postprocessOutput(W, H, inputBlob, outputBlobs)

            if m == 0:
                segmapScale1 = seg
            elif m == 1:
                segmapScale2 = seg         
            elif m == 2:
                segmapScale3 = seg
            elif m == 3:
                segmapScale4 = seg


        # Запускаем модель на горизонтально-перевернутой ориентации
        for m, scale in enumerate(self.multiplier):

            # Формируем входной блоб
            inputBlob = self.__makeInputBlob(flippedImg_f32, scale)

            # Колдуем
            outputBlobs = self.model.predict(inputBlob)
            if self.verbose:
                print( "\tProcessed flipped blob with size:", inputBlob.shape)
        
            # Делаем пост-обработку результата
            seg = self.__postprocessOutput(W, H, inputBlob, outputBlobs)
            seg_recover = self.__recoverFlippingOutput(img_f32, seg)

            if m == 0:
                segmapScale5 = seg_recover
            elif m == 1:
                segmapScale6 = seg_recover         
            elif m == 2:
                segmapScale7 = seg_recover
            elif m == 3:
                segmapScale8 = seg_recover
        
        # Формируем окончательную (усредненную) маску
        segAvg = np.zeros((H, W, self.partsNum))
        segmapA = np.maximum(segmapScale1, segmapScale2)
        segmapB = np.maximum(segmapScale4, segmapScale3)
        segmapC = np.maximum(segmapScale5, segmapScale6)
        segmapD = np.maximum(segmapScale7, segmapScale8)
        segOrig = np.maximum(segmapA, segmapB)
        segFlipped = np.maximum(segmapC, segmapD)
        segAvg = np.maximum(segOrig, segFlipped)

        return segAvg

    def __makeInputBlob(self, img_f32, scale):
        """Формируем входной блоб для модели.
        
        @param img_f32 входное изображение в формате RGB.
        @param scale коэффициент сжатия.

        @return входной блоб размерностью 1xHxWx3.
        """

        scaledImg = cv2.resize(img_f32, (0, 0), fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
        return scaledImg[np.newaxis, ...]

    def __postprocessOutput(self, W, H, inputBlob, outputBlobs):
        """Обрабатываем выход.
        
        @param W длина оригинального изображения.
        @param H ширина оригинального изображения.
        @param inputBlob входное блоб.
        @param outputBlobs выходной блоб.

        @return входной блоб размерностью HxWx7.
        """

        seg = np.squeeze(outputBlobs[0])
        seg = cv2.resize(seg, (0, 0), fx = self.stride, fy = self.stride,
                            interpolation = cv2.INTER_CUBIC)
        pad = [ 0,
                0, 
                (inputBlob.shape[1] - self.stride) % self.stride,
                (inputBlob.shape[2] - self.stride) % self.stride
            ]
        paddedImg = np.pad(inputBlob[0], ((0, pad[2]), (0, pad[3]), (0, 0)), \
            mode = 'constant', \
            constant_values = ((0, 0), (0, 0), (0, 0)))
        seg = seg[:paddedImg.shape[0] - pad[2], :paddedImg.shape[1] - pad[3], :]
        seg = cv2.resize(seg, (W, H), interpolation = cv2.INTER_CUBIC)
        return seg

    def __recoverFlippingOutput(self, oriImg, part_ori_size):
        part_ori_size = part_ori_size[:, ::-1, :]
        part_flip_size = np.zeros((oriImg.shape[0], oriImg.shape[1], self.partsNum))
        part_flip_size[:, :, self.humanPart] = part_ori_size[:, :, self.humanPart]
        return part_flip_size


def main():
    print('>>> Program started')

    # Парсим параметры
    parser = argparse.ArgumentParser(description = 'EVRAZ AI Challenge demo')
    # parser.add_argument('--gpus', metavar = 'N', type = int, default = 1)
    parser.add_argument('--model', type = str, default = './CDCL/weights/model_simulated_RGB_mgpu_scaling_append.0024.h5', help = 'path to the weights file')
    parser.add_argument('--input', type = str, default = './input', help = 'path to the folder with test images')
    parser.add_argument('--output', type = str, default = './output', help = 'path to the folder with result images')
    parser.add_argument('--scale', action = 'append', default=['1'], help = 'desired CDCL scales')
    parser.add_argument('--verbose', type = bool, default = True, help = 'print detailed info')
    args = parser.parse_args()

    if not os.path.isdir(args.input):
        print(">>> Input folder does not exist. Program finished.")
        sys.exit()
        
    if not os.path.exists(args.output):
        os.makedirs(args.output)
        
    # Готовим файл для submission
    with open('submission/dummy_json.json') as f:
        j = json.load(f)

    model = CDCL7(args) # Загружаем модель
    
    # Обрабатываем входные изображения
    Id = 1
    for filename in os.listdir(args.input):

        # Проверяем файл
        if not (filename.endswith(".png") or filename.endswith(".jpg")):
            continue

        # Загружаем изображение
        imagePath = os.path.join(args.input, filename)
        print(">>> Processing image", imagePath)
        img = cv2.imread(imagePath, cv2.IMREAD_COLOR)
        if img is None:
            print('>>> Could not read image')

        # Находим части тела
        seg = model.Infer(img)

#         # Выводим маску вероятностей
#         humanConfMask = utils.humanConfidenceMask(seg)

#         # Выводим сегментную маску
#         segMaskImg = img.copy()
#         humanMask = utils.humanMask(seg, thr)
#         utils.drawMask(segMaskImg, humanMask)

        # Выводим прямоугольники
        bboxesImg = img.copy()
        thr = 0.1
        humanBBoxes = utils.humanBBoxes(seg, thr)

        # Фильруем прямоугольники
        rgbMask = utils.humanRBGMask(seg, thr)
        filteredHumanBBoxes = []
        for bbox in humanBBoxes:
            if utils.containsHuman(rgbMask, bbox, thr):
                filteredHumanBBoxes.append(bbox)
        utils.drawBBoxes(bboxesImg, filteredHumanBBoxes)

        # Рисуем детальную картинку
        rgbMaskImg = cv2.addWeighted(rgbMask, 0.3, img, 0.7, 0)
        detailedImg = rgbMaskImg.copy()
        utils.drawBBoxes(detailedImg, filteredHumanBBoxes, (0, 0, 255), 2)

        # Сохраняем результат
        image_id = -1
        for image in j['images']:
            if image['file_name'] == filename:
                image_id = int(image['id'])
                break
        for bbox in filteredHumanBBoxes:
            area = float(bbox[2] * bbox[3])
            if area < 2:
                continue
            j['annotations'].append({
                "id": Id,
                "image_id": image_id,
                "category_id": 1,
                "segmentation": [],
                "area": area,
                "bbox": [
                    float(bbox[0]),
                    float(bbox[1]),
                    float(bbox[2]),
                    float(bbox[3])
                ],
                "iscrowd": 0,
                "attributes": {
                    "occluded": False
                }
            })
            Id += 1
        
        fname1 = '%s/%s.jpg' % (args.output, 'bbox_' + filename)
        cv2.imwrite(fname1, bboxesImg)
        fname2 = '%s/%s.jpg' % (args.output, 'rbg_' + filename)
        cv2.imwrite(fname2, rgbMaskImg)
        fname3 = '%s/%s.jpg' % (args.output, 'full_' + filename)
        cv2.imwrite(fname3, detailedImg)
        
        # Сохраняем submission
        with open("submission/submission.json", 'w') as f:
            json.dump(j, f)

    print('>>> Program successfully finished')
    
    
if __name__ == "__main__":
    main()