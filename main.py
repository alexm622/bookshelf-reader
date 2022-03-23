import fiftyone as fo
import fiftyone.brain as fob

IMAGES_DIR = "bookshelf-dataset/images"
LABELS_PATH = "bookshelf-dataset/runs/labelme2coco/dataset.json"
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
print(fo.list_datasets())

counts = dataset.count_values("ground_truth.detections.label")

print(counts)


session = fo.launch_app(dataset)
i = 0
while True:
    i += 1
