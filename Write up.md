
# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model-2.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* Behavioral Cloning.ipynb containing same code as model.py and the MSE graph

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network from NVIDA's autonomous driving team. The model includes 3 parts-image pre-processing, convolution layers and fully-connected layers.
The pre-processing part includes cropping and normalizing.
The conolution layers include RELU layers to introduce nonlinearity.
The fully-connected layers have "dropout" to reduce overffiting. 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 77-89). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (model.py line 21). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.


#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 92).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I downloaded the data from Udacity and used all 3 cameras. 
For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was based on the paper "End-to-end Deep Learning for Self-driving Cars" from NVIDA's autonomous driving team.


In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that with a low batch size-32 batches, the training set and validation set has a fluctuating low mean square error. Then I change the batch size to 128 and it fixed the problem.

To combat the overfitting, I modified the model by having drop-out layers between the fully-connected layers so the low mean square error doesn't fluctuate that much.


The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track, for example, there is a curve after the first dirt road, my vehicle failed a lot of times at that spot.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.


#### 2. Final Model Architecture


Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)
![NVIDA%20_network.PNG](attachment:NVIDA%20_network.PNG)


#### 3. Creation of the Training Set & Training Process

Here is an example image of center lane driving:
![center%20img.png](attachment:center%20img.png)

I flipped all the images and added to the images list:
![center%20img%20flip.png](attachment:center%20img%20flip.png)


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to drive back to the center when its off the right track. These images show what a recovery looks like:

![center%20img.png](attachment:center%20img.png)
![left%20img.png](attachment:left%20img.png)
![right%20img.png](attachment:right%20img.png)


After the collection process, I randomly shuffled the data set and put 20% of the data into a validation set. After flipping each image, I had 38568 number of data points for training set. I then preprocessed this data by cropping to 65 x 320.


I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 since I tried 10 epochs and it didn't improve obivously compared to 5 epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.Z
