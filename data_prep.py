
# importing libraries
print("[INFO]: importing Libraries")
from os import sep as separator
from imutils.paths import list_images
from sklearn.preprocessing import LabelBinarizer
import argparse
import numpy as np

from datasets.scripts.simple_dataset_loader import Simple_Dataset_Loader
from preprocessors.image_to_array import Image_to_Array
from preprocessors.imagenet import Imagenet
from cacher.file_cacher import File_Database

# commandline args
print("[INFO]: setting up commandline args")
ap = argparse.ArgumentParser()
ap.add_argument("--input_path", "-input", help="full directory path to dataset", type=str, required=True)
ap.add_argument("--output_path", "-output", help="full directory path to store cache file", type=str, default="./db.h5")
ap.add_argument("--dimension", "-dim", help="dimension(M, N) of data to be cached", nargs="+", type=int, default=[224, 224])
ap.add_argument("--batch_size", "-bs", help="batches of data to be cached per step", type=int, default=32)
args = vars(ap.parse_args())

batch_size = args["batch_size"]
dataset = args["input_path"]
output_path = args["output_path"]
image_dimension = list(args["dimension"])

# setting up database
print("[INFO]: setting up database")
dataset_path = list(list_images(dataset))
label = [i.split(separator)[-2] or i.split("/") for i in dataset_path]

dimension = [len(dataset_path),]
dimension.extend(image_dimension)
dimension.append(3)

db = File_Database(
    output_path=output_path,
    buffSize=80,
    dimension = dimension,
    numOfClasses=len(np.unique(label))
)

# preprocessors
print("[INFO]: loading preprocessor")
IAp = Image_to_Array()
Ip = Imagenet()
sdl = Simple_Dataset_Loader(preprocessors=[IAp, Ip])

# setting up encoder
le = LabelBinarizer().fit(label)

# store un-encoded classname/label
db.store_class_labels(le.classes_)

# loop over data in batches
for i in range(0, len(dataset_path), batch_size):
    # preprocess data and get corresponding encoded label
    batchPath = dataset_path[i : i + batch_size]
    batchImage = sdl.preprocess(img_path=batchPath, target_size=image_dimension, include_labels=False)
    batchImage = np.array(batchImage, dtype="float")
    batchLabel = label[i : i + batch_size]
    batchLabel = le.transform(batchLabel)
    
    # cache/store data
    db.add(batchImage, batchLabel)
    print(f"[INFO]: preprocessed {i + batch_size}/{len(dataset_path)}")

# close database/cache-storage
db.close()
print("[INFO]: succesfully cached data")