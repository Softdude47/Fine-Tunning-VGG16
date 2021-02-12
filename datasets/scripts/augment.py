from keras.preprocessing.image import ImageDataGenerator

class Augment:
    @staticmethod
    def preprocessor():
        # data preprocessing(augmentation)
        augmentation_function = ImageDataGenerator(
            rotation_range=30, width_shift_range=0.1,
            height_shift_range=0.1, shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
        )
        return augmentation_function