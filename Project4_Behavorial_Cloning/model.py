#!/usr/bin/env python3
"""This module fits a single model and logs results into MLflow.

Command line arguments can be used to control selected hyperparameters.
See python model.py --help for details
"""
from pathlib import Path

import click
import funkybob
from imageio import imread
import mlflow
from mlflow.tensorflow import autolog
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import tensorflow as tf
import tensorflow.keras.layers as layers


DATA_PATH = Path("~/data/udacity-project4-sim-data").expanduser()
BATCH_SIZE = 128


def _load_csvs(data_path):
    """Load CSV & correct paths from local data collection location to
    what we need here to train on GPU.
    """
    csvs = data_path.glob("*/*.csv")
    csv_header = ["center", "left", "right", "steering_angle", "throttle", "brake", "speed"]
    df = pd.concat((pd.read_csv(csv, names=csv_header) for csv in csvs))
    df = df.drop(["throttle", "brake", "speed"], axis=1)

    def _update_path(path):
        original_paths = ["/Users/jvlier/SimData", "/Users/jvlier/Udacity_Project4/data"]
        for orig_path in original_paths:
            if orig_path in path:
                return path.replace(orig_path, str(data_path))
        raise FileNotFoundError(f"Couldn't fix path: {path}")

    for col in ["left", "center", "right"]:
        df[col] = df[col].map(_update_path)

    return df


def _load_images(data_idx):
    """Load images from *center* camera only."""
    cam = "center"
    img_paths = data_idx[cam].values
    # It turns out that I was using left accidentally for v1 and v2 due to
    # an error in the csv header.
    # Assert now to be sure:
    assert all(Path(s).stem.startswith(cam) for s in img_paths)
    imgs = np.stack([imread(path) for path in img_paths])
    print(f"Loaded {len(imgs)} images")
    return imgs


def _load_data(hflip_augmentation: bool = False):
    """Load images and angles, optionally with horizontal flip."""
    data_idx = _load_csvs(DATA_PATH)
    imgs = _load_images(data_idx)
    angles = data_idx["steering_angle"].values

    if not hflip_augmentation:
        return imgs, angles

    # Apply horizontal flip augmentation
    hflip_imgs = np.array([np.fliplr(img) for img in imgs])
    hflip_angles = angles * -1

    all_imgs = np.concatenate((imgs, hflip_imgs))
    all_angles = np.concatenate((angles, hflip_angles))

    return all_imgs, all_angles


def _build_model(img_shape, *, learning_rate, dropout):
    """Build and compile the model."""
    model = tf.keras.Sequential()
    # Crop 50 pix from top, 40 from bottom:
    model.add(layers.Cropping2D(cropping=((50, 40), (0, 0)), input_shape=img_shape))
    # Centering and scaling with a Lambda layer ensures that data
    # coming in at test time can also be centered and scaled 
    # efficiently:
    model.add(layers.Lambda(lambda x: (x / 128.) - 0.5))

    pretrained_net = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        pooling="avg",  # This applies AveragePooling at the end to flatten output.
    )
    model.add(pretrained_net)
    model.add(layers.Dropout(dropout))

    # No activation function:
    model.add(layers.Dense(1))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
        loss=tf.keras.losses.mean_squared_error,
        metrics=[]
    )

    return model


def _fit(model, imgs, angles, *, epochs, lr_start):
    """Fit the model using ReduceLROnPlateau and EarlyStopping
    callbacks.
    """
    # validation_split in model.fit() picks from the end of the array
    # by default, so shuffle first to get a random split:
    imgs, angles = shuffle(imgs, angles, random_state=42)

    factor = .1
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        factor=factor,
        patience=5,
        verbose=1,
        min_lr=lr_start * factor ** 2   # Allows for 2 drops
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        verbose=1,
        patience=12,
        # The next arg causes MLflow to log metrics of restored (best) 
        # model.
        # WARNING: this only works in case early stopping is indeed
        # triggered. If it isn't, the MLflow metrics aren't reliable.
        # Workaround: call fit with a very high # of epochs in order to
        # guarantee early stopping.
        # Google search reveals that this is a known quirk, to fix use
        # ModelCheckpoint instead, with save_best_only, and
        # *always* load the best when fitting is completed.
        # But that might require explict logging to MLflow rather than 
        # the autolog I'm using now.
        restore_best_weights=True
    )

    model.fit(imgs, angles, batch_size=BATCH_SIZE, epochs=epochs, validation_split=.2,
              callbacks=[early_stop, reduce_lr])


@click.command()
@click.option("--exp", help="Name of experiment (in mlflow)")
@click.option("--epochs", default=200,
              help="Maximum number of epochs to train for (unless early stop gets triggered).")
@click.option("--lr", default=1e-2, help="Learning rate.")
@click.option("--dropout", default=.5, help="Dropout rate (fraction of units to drop).")
def main(exp, epochs, lr, dropout):
    # I use random but memorable model ids to easily tell them apart
    # in the MLflow UI:
    model_id = next(iter(funkybob.RandomNameGenerator()))
    print(f"Model id is {model_id}")

    imgs, angles = _load_data(hflip_augmentation=True)

    mlflow.set_experiment(exp)
    mlflow.start_run(run_name=model_id)
    autolog()
    mlflow.log_param("dropout", dropout)

    model = _build_model(imgs.shape[1:], learning_rate=lr, dropout=dropout)

    print(f"{model_id} starting training...")
    _fit(model, imgs, angles, epochs=epochs, lr_start=lr)

    mlflow.end_run()


if __name__ == "__main__":
    main()
