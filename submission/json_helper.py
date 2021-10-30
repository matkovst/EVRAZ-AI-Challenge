import json

with open('dummy_json.json') as fj:
    j = json.load(fj)

# search for an image
for image in j['images']:
    if image['file_name'] == 'am3_5_frame017.jpg':
#         print(image['id'])
        break

j['annotations'].append({
            "id": 1,
            "image_id": 1,
            "category_id": 1,
            "segmentation": [],
            "area": 9660.228599999999,
            "bbox": [
                838.78,
                268.37,
                59.62,
                162.03
            ],
            "iscrowd": 0,
            "attributes": {
                "occluded": False
            }
        })

with open("tst.json", 'w') as fj:
    json.dump(j, fj)