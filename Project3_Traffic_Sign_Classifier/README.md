Writeup - Traffic Sign Recognition Project
==========================================

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/sign-distribution.png "Sign Distribution"
[image2]: ./examples/random-signs.png "Random signs"
[image3]: ./examples/train-log.png "Training log"

[web1]: ./data/web-images/30kmzone-end.jpg "web image 1"
[web2]: ./data/web-images/30kmzone.jpg "web image 2"
[web3]: ./data/web-images/no_limit.jpg "web image 3"
[web4]: ./data/web-images/play_area.jpg "web image 4"
[web5]: ./data/web-images/priority.jpg "web image 5"
[web6]: ./data/web-images/yield.jpg "web image 6"

# Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing the number of signs for each class:

![alt text][image1]

Here are some random examples from the training set:

![alt text][image2]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I prefer to introduce as little complexity as possible in my pre-processing pipelines. My strategy was to only do the bare minimum: normalize the data, and then train a first model, and iterate from there. Fortunately, it turned out that I could get a >= .93 valid score without more pre-processing.

Conversion to grayscale was suggested and would be the first thing to try if performance on the validation set was too low. This makes the problem a bit simpler and reduces the number of parameters to learn. However, it might also throw away useful information.

One other step that I would have explored is to crop the images to the provided bounding boxes (and then resize/upscale again to 32 x 32), effectively removing noise around the sign, and making each sign the same size.

Lastly, I would have explored data augmentations in order to reduce overfitting:

- random +/- brightness
- random +/- contrast
- random (subtle) rotations 
- random (subtle) warps

In the end, I opted not to do it because Dropout worked quite well to reduce overfitting.


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| ReLU					|												|
| Max pooling 2x2	      	| 2x2 stride,  outputs 14x14x6 				|
| Convolution 5x5	    | 1x1 stride, valid padding, outputs 10x10x16      									|
| ReLU					|												|
| Max pooling 2x2	      	| 2x2 stride,  outputs 5x5x16 				|
| Flatten						| Outputs 400												|
| Fully connected		| Outputs 120        									|
| ReLU					|												|
| Dropout				| rate = 0.5												|
| Fully connected		| Outputs 84        									|
| ReLU					|												|
| Dropout				| rate = 0.5												|
| Fully connected		| Outputs 43        									|
| Softmax				| (Technically not part of the model as it output Logits)        									|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used the cross entropy loss function. The Adam optimizer was used with LR = 0.0003. I initially used 0.001 as used in the course, but this seemd a bit too high: loss was quite jittery after about 10-20 epochs. Alternatively, I could also have dropped the LR at this point, or continuously using a decay, but again: that wasn't needed to reach a good score and it trained sufficiently fast on a K80 with LR = 0.0003. The betas for Momentum/RMSProp remained at the defaults.

I left batch size at the default of 128. This could have gone a bit higher (enough memory on a 12 GB K80), but 128 is already quite high and I don't want to remove even more stochasticity from the optimization process.

I found empirically that training for approx. 30 epochs was about right. The validation loss was pleateauing at this point, but train loss was still improving. Training longer results in overfitting: train loss keeps on going down but valid loss goes up.

I ended up taking the model checkpoint at epoch 26, but have trained it until 40.

Loss and accuracy per epoch:

![alt text][train-log] 

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:

* training set accuracy of .998
* validation set accuracy of .961
* test set accuracy of .942

I based my implementation on the LeNet network we covered in class, as suggested in the project instructions. It is a pretty simple and shallow CNN, but traffic signs are also quite simple. This made me believe that it could be a good fit for traffic signs. We don't need to be able to detect very complicated textures or patterns. In addition, a simple/shallow neural net like this would also be actually implementable in a car for real-time predictions (in contrary to a heavy ResNet152, for instance).

The changes I made are the following:

- Using all RGB channels instead of grayscale.
- Added dropout to the two fully connected layers at the end. I noticed a bit of overfitting and this compensates for that nicely.
- Changed the 10 hardcoded output classes to `n_classes`

The training accuracy is very high, meaning that the model has the capacity to detect (or at least: memorize) these traffic signs very well. The validation score is a bit lower, meaining that it could generalize to unseen data a bit better. I believe that adding more training data or augmented training data could help in this aspect. The test score is a bit lower than the valid score, which makes sense, because the valid score informed some architectural and hyperparameter choices (e.g. dropout rate) during development of the model. The test score of .942 seems still a bit too low to me for real-life deployments, and I would recommend a bit more work on this before continuing to production stage.
 
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are six German traffic signs that I found on the web:

![alt text][web1] ![alt text][web2] ![alt text][web3] 
![alt text][web4] ![alt text][web5] ![alt text][web6]

As I realized later, the 30 km zone signs and the play area sign are not in the dataset. Still, it is interesting to see how the model handles these. I expect no problems for the remainder, but I did zoom in the third image a bit ("End of all speed and passing limits")  because it's quite far away compared to the training data.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

![examples/web-pred.png]

The model was able to predict 3 out of 6 traffic signs correctly, an accuracy of 50%.

However taking into account that 3 of these traffic signs don't even exist in the training dataset, the accuracy on the ones that it knows is 3/3 = 100%.
Curiously, it almost got the "End of 30 km zone" sign right. This is probably due to the gray diagonal lines.

Of course, no conclusions can be drawn on a sample of 3 or 6.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions is located in the very last code cell of the Notebook. I am re-using the predictions made in the last step and doing the softmax and selections with scipy and numpy.

For the end 30 km zone sign, (not in the dataset), the model is seems to allocate weight to two signs, the highest of which is quite similar:

The top five soft max probabilities were:

|   ClassId | SignName                            |        prob |
|----------:|:------------------------------------|------------:|
|        32 | End of all speed and passing limits | 0.681861    |
|        15 | No vehicles                         | 0.289459    |
|        41 | End of no passing                   | 0.0200161   |
|        38 | Keep right                          | 0.00468865  |
|        12 | Priority road                       | 0.000784041 |

The start 30 km zone sign is, understandably, a bit of a disaster:

|   ClassId | SignName                                           |      prob |
|----------:|:---------------------------------------------------|----------:|
|        11 | Right-of-way at the next intersection              | 0.348971  |
|        12 | Priority road                                      | 0.167663  |
|        13 | Yield                                              | 0.104354  |
|        41 | End of no passing                                  | 0.0996327 |
|        42 | End of no passing by vehicles over 3.5 metric tons | 0.0556152 |

The no limit / end of passing restriction sign gets detected with very high confidence:

|   ClassId | SignName                            |        prob |
|----------:|:------------------------------------|------------:|
|        32 | End of all speed and passing limits | 0.999547    |
|        41 | End of no passing                   | 0.000398194 |
|         3 | Speed limit (60km/h)                | 2.97617e-05 |
|         6 | End of speed limit (80km/h)         | 1.20205e-05 |
|         1 | Speed limit (30km/h)                | 5.16992e-06 |

For the play area sign - also not in the original dataset - a totally different sign gets detected. Worryingly, it seems quite certain about this. This shows the limitation of these kind of models in handling unknown/new signs:

|   ClassId | SignName             |        prob |
|----------:|:---------------------|------------:|
|        40 | Roundabout mandatory | 0.979262    |
|        38 | Keep right           | 0.016378    |
|        39 | Keep left            | 0.00427291  |
|        15 | No vehicles          | 2.94031e-05 |
|        37 | Go straight or left  | 1.86616e-05 |

Finally two very good predictions for signs that are actually in the dataset: yield and priority, respecitvely:

|   ClassId | SignName                                           |        prob |
|----------:|:---------------------------------------------------|------------:|
|        12 | Priority road                                      | 1           |
|        25 | Road work                                          | 4.27642e-16 |
|        42 | End of no passing by vehicles over 3.5 metric tons | 1.845e-16   |
|        26 | Traffic signals                                    | 6.71081e-17 |
|        10 | No passing for vehicles over 3.5 metric tons       | 2.3878e-17  |

|   ClassId | SignName      |        prob |
|----------:|:--------------|------------:|
|        13 | Yield         | 1           |
|        12 | Priority road | 2.76869e-17 |
|        15 | No vehicles   | 1.99005e-20 |
|        35 | Ahead only    | 1.219e-20   |
|        25 | Road work     | 5.5845e-21  |

These signs have a unique and distinctive appearance, so it makes a lot of sense that the model is very confident about its predictions.
