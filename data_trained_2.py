import os
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def load_dataset(path):
    train = pd.read_csv(os.path.join(path, 'train.csv'))
    test = pd.read_csv(os.path.join(path, 'test.csv'))

    Y_train = train["label"]
    X_train = train.drop(labels=["label"], axis=1)
    X_train = X_train / 255.0
    X_test = test / 255.0
    X_train = X_train.values.reshape(-1, 150, 150, 1)
    X_test = X_test.values.reshape(-1, 150, 150, 1)


    Y_train = to_categorical(Y_train, num_classes=15)

    X_train, X_val, Y_train, Y_val = train_test_split(
        X_train, Y_train, test_size=0.1, random_state=42)

    return X_train, X_val, Y_train, Y_val


def build_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(150, 150, 1)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dense(15, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def train_model(model, X_train, X_val, Y_train, Y_val):
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.10,
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    train_generator = datagen.flow(X_train, Y_train, batch_size=32)
    val_generator = datagen.flow(X_val, Y_val, batch_size=32)
    model.fit(train_generator, epochs=20, validation_data=val_generator)


def save_model(model, model_path):
    model.save(model_path)
    print(f"Model saved to {model_path}")


if __name__ == "__main__":
    dataset_path = 'D:/barovinh/Python/Dataset'  # ĐƯỜNG DẪN ĐẾN THƯ MỤC CHỨA TẬP DỮ LIỆU
    model_path = 'D:/barovinh/Python/Dataset/model.h5'  # ĐƯỜNG DẪN ĐỂ LƯU MÔ HÌNH

    X_train, X_val, Y_train, Y_val = load_dataset(dataset_path)
    model = build_model()
    train_model(model, X_train, X_val, Y_train, Y_val)
    save_model(model, model_path)
