# EVRAZ-AI-Challenge
Данный репозиторий содержит финальное решение для хакатона EVRAZ AI Challenge (Трек 2). </br>
За основу решения взята модель [CDCL7](https://github.com/kevinlin311tw/CDCL-human-part-segmentation), которая сегментирует части тела человека. Сам алгоритм следующий:
1. Находим части тела
2. Формируем из них бинарную маску
3. Находим контуры из этой маски и обрамляем их прямоугольниками

## Зависимости
Все зависимости указаны в файле docker/environment.yaml.

## Установка через conda
```bash
conda env create -f docker/environment.yaml
conda activate cdcl
```

## Установка через Docker
Проект можно запустить в докере с помощью команд

```bash
docker build --tag evraz_matkovst:v1 docker/

docker run --volume ~/output:/home/EVRAZ-AI-Challenge/output -it evraz_matkovst:v1 bash
```

## Запуск
```bash
python demo.py --input input/test/ --output output/test
```

## References
<a id="1">[1]</a> 
Lin, Kevin and Wang, Lijuan and Luo, Kun and Chen, Yinpeng and Liu, Zicheng and Sun, Ming-Ting (2020). 
Cross-Domain Complementary Learning Using Pose for Multi-Person Part Segmentation. 
IEEE Transactions on Circuits and Systems for Video Technology.