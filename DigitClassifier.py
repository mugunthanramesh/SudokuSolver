from tensorflow.keras.layers import Conv2D,MaxPool2D,Flatten,Dense,Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist

class DigitsClassifier:

    def __init__(self):
        self.isTrained = False
        self.isCompiled = False
        self.model = Sequential()

    def compileModel(self):
        if(self.isCompiled and self.isTrained):
            print("Model is already Compiled and Trained :)")
            return

        if(self.isCompiled):
            print("Model is ready to Train \n Starting training now...")
            self.trainModel()
            return

        self.model.add(Conv2D(32,(5,5), input_shape=(28, 28, 1), activation="relu"))
        self.model.add(MaxPool2D(pool_size = (2,2)))
        self.model.add(Conv2D(32,(3,3), activation="relu"))
        self.model.add(MaxPool2D(pool_size = (2,2)))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation="relu"))
        self.model.add(Dense(64, activation='relu'))
        self.model.add(Dense(10, activation="softmax"))

        self.model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
        self.isCompiled = True

    def trainModel(self):
        if not self.isCompiled:
            print("Model need to be build and compiled first \n Compiling now.. :)")
            self.compileModel()

        if self.isTrained:
            prompt = input("Model already trained. Are you sure want to train again(Y/N) ?")
            if(prompt == "N" or prompt == "No"):
                return

        (x_train, y_train), (x_test, y_test) = mnist.load_data()
        x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
        x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
        x_train = x_train / 255.0
        x_test = x_test / 255.0

        self.model.fit(x_train, y_train, epochs=5)
        loss, accuracy = self.model.evaluate(x_test, y_test)

        print("Model Accuracy :" , accuracy, "\nModel Loss :", loss)
        self.isTrained = True

    def saveModel(self):
        if not self.isCompiled:
            print("Model need to be compiled and trained before saving. \n Compiling Now...")
            self.compileModel()
            print("Training Now...")
            self.trainModel()

        if not self.isTrained:
            print("Model must be trained before saving. \n Training now...")
            self.trainModel()

        self.model.save("model.hdf5", overwrite=True)


if __name__ == "__main__":
    classifier = DigitsClassifier()
    classifier.saveModel()