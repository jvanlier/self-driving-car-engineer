#!/usr/bin/env python3
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as layers
from imageio import imread
from sklearn.utils import shuffle
import click


DATA_PATH = Path("~/data/udacity-project4-sim-data").expanduser()
BATCH_SIZE = 128


def _determine_model_path(model_id):
    model_path = None
    for i in range(1, 100):
        model_path_candidate = Path("models") / f"{model_id}{i:02d}"
        if not model_path_candidate.exists():
            model_path = model_path_candidate
            model_path.mkdir(parents=True)
            print(f"Storing results in {model_path}")
            break
    if not model_path:
        raise NotADirectoryError(f"Unable to determine directory for model {model_id}")
    return model_path


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


def _build_model(img_shape, *, learning_rate, dropout):
    model = tf.keras.Sequential()
    model.add(layers.Lambda(lambda x: (x / 128.) - 0.5, input_shape=img_shape))

    pretrained_net = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        pooling="avg",  # This applies AveragePooling at the end to flatten outputk.
    )
    model.add(pretrained_net)
    # model.add(layers.Dense(128, activation="relu"))
    model.add(layers.Dropout(dropout))

    # No activation function:
    model.add(layers.Dense(1))

    # print(model.summary())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        loss=tf.keras.losses.mean_squared_error,
        metrics=[]
    )

    return model


def _fit(model, imgs, angles, model_path: Path, *, epochs):
    # validation_split in fit() picks from the end of the array by default, so shuffle first to get
    # a random split:
    imgs, angles = shuffle(imgs, angles, random_state=42)

    save_ckpt = tf.keras.callbacks.ModelCheckpoint(
        str((model_path / "model.{epoch:02d}-{val_loss:.4f}.hdf5")),
        save_best_only=True,
        verbose=1
    )
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=.2,
        patience=5,
        verbose=1,
        min_lr=4e-5  # Allows for 2 drops when starting with 1e-3 and factor = .2
    )
    early_stop = tf.keras.callbacks.EarlyStopping(verbose=1, patience=10)

    model.fit(imgs, angles, batch_size=BATCH_SIZE, epochs=epochs, validation_split=.2,
              callbacks=[
                  early_stop,
                  reduce_lr,
                  save_ckpt
              ])


@click.command()
@click.option("--epochs", default=30,
              help="Maximum number of epochs to train for (unless early stop gets triggered).")
@click.option("--lr", default=1e-3, help="Learning rate.")
@click.option("--dropout", default=.5, help="Dropout rate (fraction of units to drop).")
def main(epochs, lr, dropout):
    model_id_prefix = f"model_maxepochs-{epochs}_lr-{lr}_dropout-{dropout}_v"
    model_path = _determine_model_path(model_id_prefix)
    # model_id is the same as model_id_prefix, but with double-digit version number, e.g. v01 - v99 suffix:
    model_id = model_path.name
    print(f"Model id is {model_id}")

    data_idx = _load_csvs(DATA_PATH)
    imgs = _load_images(data_idx)
    angles = data_idx["steering_angle"].values

    model = _build_model(imgs.shape[1:], learning_rate=lr, dropout=dropout)
    _fit(model, imgs, angles, model_path, epochs=epochs)


if __name__ == "__main__":
    main()
