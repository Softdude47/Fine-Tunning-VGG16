# imports libraries
from tensorflow.keras.applications import VGG16 as Model
import argparse

# set up commandline args
ap = argparse.ArgumentParser()
ap.add_argument("--include_top", "-i", help="wether or not to include the final layer / top of the CNN", type=int, required=True)
args = vars(ap.parse_args())

# load model
model = Model(weights="imagenet", include_top=args["include_top"] > 0)

# loop over model layers
for (i, name) in enumerate(model.layers):
    # print layer index and name separated
    # by a horizontal tab
    print(f"{i}\t{name.__class__.__name__}")