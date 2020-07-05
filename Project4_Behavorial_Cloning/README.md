# **Behavioral Cloning** 
Writeup - Behavorial Cloning Project
====================================

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[mlflow]: ./writeup-imgs/mlflow.png "MLflow example image"
[center-lane]: ./writeup-imgs/center-lane.png "Example of center line driving"
[recovery1]: ./writeup-imgs/recovery1.png "Example of recovery 1"
[recovery2]: ./writeup-imgs/recovery2.png "Example of recovery 2"
[recovery3]: ./writeup-imgs/recovery3.png "Example of recovery 3"
[flip1]: ./writeup-imgs/flip1.jpg "Example of flip 1"
[flip2]: ./writeup-imgs/flip2.jpg "Example of flip 2"
[mlflow-final]: ./writeup-imgs/mlflow-final.png "MLflow final experiment"
[tensorboard]: ./writeup-imgs/tensorboard.png "Tensorboard"

# Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `README.md`: this writeup, summarizing the results, with images in `writeup-imgs`
* `drive.py`: for driving the car in autonomous mode
* `model.py`: containing the script to create and train the model
* `model.h5`: containing a trained convolution neural network 
* `requirements.txt`: Python packages to recreate the environment
* `train-many.sh`: A bash script which runs model.py multiple times with different arguments (for the hyperparameter search)
* `video.mp4`: video with the vehicle completing a full lap

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 

```sh
python drive.py model.h5
```

Note: the data was originally collected using "Fastest" Graphics Quality. To make the environment look similar as to what the model was exposed to during training, ensure to pick Fastest when testing. The resolution does not matter.

#### 3. Submission code is usable and readable

The `model.py` file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a pre-trained MobileNetV2 Convolutional Neural Network. This is an efficient model that can do infererence on low-end or power constrained hardware, such as what might be found in a car. In addition, this model is not sensitive to input size (in contrast to e.g. VGG16), which allowed me accept the widescreen images without trouble, and made it easy to experiment with crops. On top of the MobileNetV2, after the final AveragePooling, I added a Dropout layer to control for overfitting and a single output node (without activation function, since this is a regression problem).

Before the data enters the MobileNetV2, I added a Cropping2D layer to remove unneccesary pixels from the top and the bottom of the image, and used a Lambda layer to rescale the image. Placing the pre-processing inside the network allows for seamless inference in the simulator's autonomous mode.

The model definition can be found in `model.py`, function `_build_model(...)`, lines 79 - 98.

#### 2. Attempts to reduce overfitting in the model
MobileNetV2 employs Dropout and batch normalization to control overfitting. The Dropout rate was left at the default in Keras' implementation (0.001).

In addition to that, due to dataset being relatively small, I added another Dropout layer just before the output layer to further reduce overfitting (`model.py` line 95).
I also added horizontal flips to augment the data (`model.py` lines 69 - 76). The Dropout rate in the final model was 0.5.

The model was trained and validated on different data sets to ensure that the model was not overfitting (in `model.py` see `validation_split=.2` on line 141). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

The final measure that reduces overfitting is early stopping: I automatically stop training when a plateau has been reached, and use the weights that correspond to the lowest training loss (see the `EarlyStopping` callback on lines 124 - 139 in `model.py`).

#### 3. Model parameter tuning

I use an Adam optimizer, but I still find that it is valuable to use an appropriate learning rate (LR): a higher LR leads to faster training initially, as long as it isn't so high that it causes the loss to diverge. This, combined with dropping the LR as a plateau is reached (also called learning rate annealing), leads to finding a better model in less time.

To find an optimal initial LR and rate for the final Dropout layer, I ran multiple experiments and logged the results in MLflow. Within MLflow, I was able to easily compare diferent runs and choose the values that yielded the lowest validation loss. Using this method, I settled on an initial learning rate 0.01 and dropout of .5. Note that the LR gets dropped automatically by a factor 10, at most twice, when a plateau has been encountered (see the `ReduceLROnPlateau` callback on lines 118 - 123 in `model.py`).

Example of what this looks like in MLflow (not the final model):

![MLflow example image][mlflow]

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of:

- center lane driving: 2 laps
- recovering from the left and right sides of the road: 2 laps
- center lane driving, reverse direction: 1 lap
- various specific examples to counteract bad behavior seen in simulator with earlier versions of the model

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to start from an existing lightweight Convolutional Neural Network. A large advantage of using an existing Neural Network architecture is the ability to leverage peer reviewed research, and the ability to load pre-trained weights which significantly reduces training time.

My first step was to start out as simple as possible: just this network, the Lambda layer to rescale the pixel values, and a single output node without activation function.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat overfitting, I added a Dropout layer just before the output layer.

After this, the difference in loss between train and validation was much smaller, and both losses were quite low in general (< 0.01). 
Next, I ran the simulator and noticed it fell off the track in the first dirt turn. To counteract this, I added more data from this specific turn (approx 10 examples) and tried again.

Unfortunately, after retraining, the car would drop off at a different turn. I now made several changes:

- Crop the top and bottom part of the image to improve bias
- Horizontally flip the images and angles to improve variance (reduce overfitting)

Retraining after these tweaks led to the final model in this submission: it successfully drove around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (`model.py` lines 79 - 98) consisted of a pre-trained MobileNetV2 architecture (with ImageNet weights). The base of that model was followed by a Dropout layer and a single output node (no activation function). Cropping and centering & scaling are done inside the model. The model is summarized as follows:

```
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
cropping2d (Cropping2D)      (None, 70, 320, 3)        0
_________________________________________________________________
lambda (Lambda)              (None, 70, 320, 3)        0
_________________________________________________________________
mobilenetv2_1.00_224 (Model) (None, 1280)              2257984
_________________________________________________________________
dropout (Dropout)            (None, 1280)              0
_________________________________________________________________
dense (Dense)                (None, 1)                 1281
=================================================================
Total params: 2,259,265
Trainable params: 2,225,153
Non-trainable params: 34,112
```

Where `mobilenetv2_1.00_224` corresponds to the Keras implementation of MobileNetV2 [here](https://keras.io/api/applications/mobilenet/). [Link to original paper](https://arxiv.org/abs/1801.04381).

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an screenshot of the simulator showing center lane driving:

![Example of center line driving][center-lane]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the center of the road after veering of course. These images show the point at which a recovery example recording would typically start:

![Example of recovery 1][recovery1]
![Example of recovery 2][recovery2]
![Example of recovery 3][recovery3]

Then I repeated the lap but in reverse direction in order for the model to generalize better, because the track is biased towards left turns in the default direction.

Finally, I added more data of center line driving and recovery in places that were observed to be problematic in various autonomous runs in the similar with earlier versions of the model.

I did not use track 2 at all.

To augment the data set, I also flipped images and angles thinking that this would improve generalization, as mentioned earlier. For example, here is an image that has then been flipped:

![Example of flip 1][flip1]
![Example of flip 2][flip2]

After the collection process, I had 9.500 of data points (pre augmentation). The data was not preprocessed since this was handled inside the fist layer of the network (centering, scaling and cropping.)


I finally randomly shuffled the data set and put 20 % of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over- or underfitting. The training approach involves stepwise learning rate annealing, early stopping + using the weights that minimized the validation loss, and exploring the hyperparameter space with MLflow. This was explained in detail above. The final model had a dropout of 0.5, optimal starting learning rate of 0.01, and had the lowest validation loss of 0.003 at 63 epochs. Training was terminated at 75 epochs (at which point the learning rate was dropped to 0.0001).

![MLflow final experiment][mlflow-final]

![tensorboard][tensorboard]

(This experiment was ended early to save costs, because the very first model that was trained was already capable of going around the track.)

