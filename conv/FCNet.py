from keras.layers.core import Dense, Dropout, Flatten

class FCNet:
    @staticmethod
    def build(baseModel, units, classes, activation="relu"):
        # the base and then, fully connected layer
        inputs = baseModel.output
        headModel = Flatten(name="flatten")(inputs)
        headModel = Dense(units=units, activation=activation)(headModel)
        headModel = Dropout(0.5)(headModel)
        
        # final layer
        headModel = Dense(units=classes, activation="softmax")(headModel)
        return headModel