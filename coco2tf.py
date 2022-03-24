import fiftyone as fo
import fiftyone.brain as fob
training_data = True
if training_data:
    t_or_v = "training-data"
else:
    t_or_v = "validation data"

IMAGES_DIR = "workspace/bookshelf-dataset/"+ t_or_v + "/images"
LABELS_PATH = "workspace/bookshelf-dataset/"+ t_or_v + "/runs/labelme2coco/dataset.json"
if "books" not in fo.list_datasets():
    dataset = fo.Dataset.from_dir(
        dataset_type=fo.types.dataset_types.COCODetectionDataset,
        data_path=IMAGES_DIR,
        labels_path=LABELS_PATH,
        label_field="",
        name="books",
    )
else:
    dataset = fo.load_dataset("books")
#export the dataset

dataset.export(
    export_dir="workspace/bookshelf-dataset/"+ t_or_v + "/runs/as-tfrecord",
    dataset_type=fo.types.dataset_types.TFObjectDetectionDataset
)
'''
session = fo.launch_app(dataset)
i = 0
while True:
    i += 1
'''