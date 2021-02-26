# import necessary
print("[INFO]: importing Libraries")
from datasets.scripts.augment import Augment
from conv.FCNet import FCNet


from argparse import ArgumentParser
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.applications import VGG16
from tensorflow.keras import Model, Input
from h5py import File

# commandline arguments
ap = ArgumentParser()
ap.add_argument("--output_path", "-output", help="path(filename included) to save fine tunned model", type=str, default="./model.h5")
ap.add_argument("--train_size", "-train", help="decimal value specifying train data percentage", type=float, default=0.8)
ap.add_argument("--input_path", "-input", help="path(filename included) dataset", type=str, required=True)
ap.add_argument("--batch_size", "-bs", help="batch size of training", type=int, default=32)
ap.add_argument("--learning_rate", "-lr", help="learn rate optimizer", type=float, default=0.0001)
ap.add_argument("--classes", "-c", help="number of classes", type=int, default=17)
ap.add_argument("--epochs", "-e", help="number of epochs", type=int, default=32)
args = vars(ap.parse_args())


epochs = args["epochs"]
classes = args["classes"]
batch_size = args["batch_size"]
model_path = args["output_path"]
dataset_path = args["input_path"]
learning_rate = args["learning_rate"]

# load dataset 
db = File(dataset_path, mode="r")
idx = int(len(db["features"]) * args["train_size"])

# split dataset
x_train = db["features"][ : idx]
y_train = db["labels"][ : idx]

x_test= db["features"][idx : ]
y_test= db["labels"][idx : ]


''' Constructing Model '''
# load pre-train model and cuts off the head(final layer)
print("[INFO]: loading pre-trained model")
base_model = VGG16(
    weights="imagenet",
    include_top=False,
    input_tensor=Input(shape=(224,224,3))
)

# creates another model by attaching new head to the pre-trained model
print("[INFO]: creating new model")
head_model = FCNet.build(baseModel=base_model, units=512, classes=classes, activation="relu")
model = Model(inputs=base_model.input, outputs=head_model)

# freezing the body of model
# thereby making only the head trainable
print("[INFO]: freezing body of new model")
for i in base_model.layers:
    i.trainable = False

# compiling model
print("[INFO]: compiling model")
optimizer = RMSprop(lr=learning_rate)
model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

# trainig model / warming model head
print("[INFO]: warming up fully connected layers")
gen = Augment.preprocessor()
model.fit_generator(gen.flow(
    x=x_train, y=y_train, batch_size=batch_size
    ), steps_per_epoch=len(x_train)//epochs, epochs=epochs, validation_data=(x_test, y_test)
)

# save model
model.save(model_path)
