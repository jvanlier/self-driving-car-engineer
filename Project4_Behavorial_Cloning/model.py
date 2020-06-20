from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
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
    model = tf.keras.applications.NASNetMobile(
        #input_shape=img_shape,
        # I think img_shape should work as input size, but it doesn't - not sure why.
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    # Shape now is (None, 1056), so 2D (first dim is batch size) 
    print(model.summary())

    #l = tf.keras.layers
    #model.add(l.


def main():
    data_idx = _load_csvs(DATA_PATH)
    imgs = _load_images(data_idx)
    angles = data_idx["steering_angle"].values

    # imgs.shape: (4600, 160, 320, 3)
    # angles.shape: (4600, )

    _build_model(imgs.shape[1:])


if __name__ == "__main__":
    main()
