# EVRAZ-AI-Challenge
![](https://github.com/matkovst/EVRAZ-AI-Challenge/blob/master/teaser.jpg)
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
bash fetch_data.sh
```

## Установка через Docker
Проект можно запустить в докере с помощью команд

```bash
docker build --tag evraz_matkovst:v1 docker/

docker run --volume ./output:/home/EVRAZ-AI-Challenge/output -it evraz_matkovst:v1 bash
```

## Запуск
Положите тестовые изображения в папку ./input и выполните команду
```bash
python demo.py --input input/ --output output/
```
Обработанные изображения будут сохранены в ./output, JSON с аннотациями будет сохранен в submission/submission.json.

## References
<a id="1">[1]</a> 
Lin, Kevin and Wang, Lijuan and Luo, Kun and Chen, Yinpeng and Liu, Zicheng and Sun, Ming-Ting (2020). 
Cross-Domain Complementary Learning Using Pose for Multi-Person Part Segmentation. 
IEEE Transactions on Circuits and Systems for Video Technology.