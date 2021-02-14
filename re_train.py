# import necessary
print("[INFO]: importing Libraries")
from datasets.scripts.augment import Augment

from argparse import ArgumentParser
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
from keras.models import load_model
from keras.optimizers import SGD
from h5py import File
from numpy import arange, unique


# commandline arguments
print("[INFO]: setting up commandline args")
ap = ArgumentParser()
ap.add_argument("--model_path", "-output", help="path(filename included) to save fine tunned model", type=str, default="./model.h5")
ap.add_argument("--train_size", "-train", help="decimal value specifying train data percentage", type=int, default=0.8)
ap.add_argument("--input_path", "-input", help="path(filename included) dataset", type=str, required=True)
ap.add_argument("--batch_size", "-bs", help="batch size of training", type=int, default=32)
ap.add_argument("--learning_rate", "-lr", help="learn rate optimizer", type=int, default=0.0001)
ap.add_argument("--epochs", "-e", help="number of epochs", type=int, default=32)
args = vars(ap.parse_args())


epochs = args["epochs"]
batch_size = args["batch_size"]
model_path = args["model_path"]
dataset_path = args["input_path"]
learning_rate = args["learning_rate"]

# load dataset 
print("[INFO]: loading dataset")
db = File(dataset_path, mode="r")
idx = int(len(db["features"]) * args["train_size"])
class_names = db["class_names"]

# split dataset
print("[INFO]: splitting dataset")
x_train = db["features"][ : idx]
y_train = db["labels"][ : idx]

x_test= db["features"][idx : ]
y_test= db["labels"][idx : ]

# data image generator
aug = Augment.preprocessor()

# create model's optimizer
print("[INFO]: configuring optimizer")
sgd = SGD(lr=learning_rate)

# load model from path
print("[INFO]: loading model")
model = load_model(model_path)

# compile model with the new optimizer
print("[INFO]: compiling model")
model.compile(sgd, loss="categorical_crossentropy", metrics=["accurary"])

# train model with augmented data from Image data generator
# and then, obtain the train/test history
print("[INFO]: training and testing model")
history = model.fit_generator(
    aug.flow(
        x=x_train, y=y_train, batch_size=batch_size
    ),
    steps_per_epoch=len(x_train)//epochs,
    epochs=epochs,
    validation_data=(x_test, y_test)
)

# getting model classification ability info
pred = model.predict(x_test, batch_size=batch_size)
report = classification_report(y_true = y_test.argmax(axis=1), y_pred = pred.argmax(axis=1), target_names=class_names)
print(report)

print("[INFO]: plotting model history")
plt.plot(arange(0, epochs), history["accuracy"])
plt.plot(arange(0, epochs), history["val_accuracy"])
plt.plot(arange(0, epochs), history["loss"])
plt.plot(arange(0, epochs), history["val_loss"])
plt.xlabel("epochs")
plt.ylabel("loss/accuracy")
plt.legend()
plt.show()