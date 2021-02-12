# Fine Tunning VGG16

## Script

### data_prep.py

* info:
  * caches and preproces dataset to be used in training and testing the model
* arguments:
  * --input_path: path to dataset
  * --output_path: full path(filename includede) to save the preprocessed and cached dataset
  * --dimension: dimension for any given feature sample of the dataset
  * --batch_size: number of feature and label to be preprocessed and cached the same time

### fine_tunning.py

* info:
  * loads the preprocessed/cached dataset and train a fine tunned VGG16 model
* arguments:
  * --input_path[_required_]: path to cached/preprocessed dataset
  * --output_path[_optional_] : full path(filename includede) to save the train model
  * --epochs[_optional_]: number of training epochs
  * --classes[_optional_] : number of classes in dataset
  * --train_size[_optional_] : fraction of dataset to be used to training
  * --learning_rate[_optional_] : learning rate of optimizer used in training model
  * --batch_size[_optional_] : number of feature and label samples to train model per each steps
