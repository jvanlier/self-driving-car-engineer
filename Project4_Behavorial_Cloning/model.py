from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
from imageio import imread


DATA_PATH = Path("~/data/udacity-project4-sim-data").expanduser()


def _load_csvs(data_path):
    csvs = data_path.glob("*/*.csv")
    csv_header = ["left", "center", "right", "steering_angle", "throttle", "brake", "speed"]
    df = pd.concat((pd.read_csv(csv, names=csv_header) for csv in csvs))
    df = df.drop(["throttle", "brake", "speed"], axis=1)

    def update_path(path):
        return path.replace("/Users/jvlier/SimData", str(DATA_PATH))

    for col in ["left", "center", "right"]:
        df[col] = df[col].map(update_path)

    return df


def _load_images(data_idx):
    img_paths = data_idx["center"].values
    imgs = np.stack([imread(path) for path in img_paths])
    return imgs


def _build_model(img_shape):
    model = tf.keras.Sequential()
    model.add(layers.Lambda(lambda x: (x / 255.0) - 0.5, input_shape=img_shape))

    pretrained_net = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        pooling="avg",  # This applies AveragePooling at the end to flatten outputk.
    )
    model.add(pretrained_net)
    # model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dense(1, activation="relu"))

    # print(model.summary())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=1e-3),
        loss=tf.keras.losses.mean_squared_error,
        metrics=[]
    )

    return model


def _fit(model, imgs, angles):
    history = model.fit(imgs, angles, batch_size=128, epochs=10)


def main():
    data_idx = _load_csvs(DATA_PATH)
    imgs = _load_images(data_idx)
    angles = data_idx["steering_angle"].values

    # imgs.shape: (4600, 160, 320, 3)
    # angles.shape: (4600, )

    model = _build_model(imgs.shape[1:])
    _fit(model, imgs, angles)


if __name__ == "__main__":
    main()
