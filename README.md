# Deepfake-Detection

Deepfake is a type of image or video forgery technique used to invade privacy, spread misinformation, that masks the truth using advanced technologies such as artificial intelligence and deep learning. With the advancement in the deep learning field and availability of data, these deepfakes looks so real that they are unable to detect by human as well as traditional detection methods.


![rsz_screenshot_66](https://user-images.githubusercontent.com/28837542/120933350-68b80e00-c717-11eb-9f04-03493fe30d4e.jpg)


In this project, we have implemented a Capsule + LSTM based model for detecting deepfake, which focuses on detecting spatio-temporal inconsistencies across frames.

This model is an extension of the the CapsuleNet model, which is proposed by Nguyen et al. in his paper “Use of a Capsule Network to detect fake images and videos”. CapsuleNet model focuses on detecting only the spatial artifacts within a single frame.

### Dataset

For comparison and evaluation of the model we have used Celeb-DF(v1) and FaceForensics++ dataset.

Links for the datasets:

Celeb-DF(v1) : http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html

FaceForensics++ : https://github.com/ondyari/FaceForensics/tree/master/dataset


### Train-Test Split

Training data(70%), Testing Data(15%) and Validation data(15%), which is used to train our model as well as evaluating its performance.


### Preprocessing

The steps for pre-processing are:

#### 1) Extracting Frames:
Split the video into a set of frames.
Since, each video in CelebDF datatset is of around 10-15 seconds and the frame
rate is 30-frame per second, so there will be around 300 frames for each video, and
if we use this number of frames for every video, then it will become very
computationally expensive. To deal with this, I have selected 10 frames from each
video in which all frames are captured at equal intervals. (Since input video is of
around 10-sec means at every 1-sec interval we capture a frame).
Similar type of operation is performed on the FaceForensics++ dataset, even if the
video length is not constant throughout all the videos. The only difference is that
some more number of frames (>10) is captured for later preprocessing stages so
that from those 10 frames are captured which will be reliable for training as well
as testing.

#### 2) Face Detection:
Since, we have to detect deepfake, so we have to mainly focus on the face area of a
person in the video, for that we run a face detection algorithm on the frames we
collected from the videos. For detecting faces in an image, different face
recognition methods are used like Haar cascade classifier, dlib , and MTCNN
in which the MTCNN performs the best , so MTCNN is used for the face detection
purpose.
In the FaceForensics++ dataset, in many videos, the person in the video is coming
from a distance, in that case the face detected from that frame is not very clear, and
the bounding box returned by the MTCNN is small in that frame as compared
when that person is finally appeared in front of the camera (face detected is very
clear). So, we sort our frames according to the bounding box area so that those
frames in which the face detected is not clear will be discarded and will not be
used for training as well as testing purposes.

#### 3) Padding:
The face detection algorithm selects only the very face area and neglects the
surrounding which will not be good for the model in order to predict as it will be
difficult to capture artifacts around the face area, so we add some padding area to
the bounding box of the face given by face detection algorithm.

#### 4) Resizing:
After getting the faces from each frame of a video, we resize each image.
We have used size of 128x128, as it is an even number and also large enough to give
information for the detection of deepfake.

#### 5) Normalization:
Before passing the image data to our model we normalize the images.
Normalizing images is an important step which ensures that each input pixels will have
a similar data distribution.
It is done by subtracting mean from each pixel and dividing the outcome by standard
deviation. We have used standard deviation = (0.229, 0.224, 0.225) and mean = (0.485, 0.456,0.406) for normalization.



### Pipeline

![image1](https://user-images.githubusercontent.com/28837542/120933314-2f7f9e00-c717-11eb-9487-f5a068a78230.jpg)



### Proposed Model

The capsule Net model proposed by Nguyen et al. is a model which exploits the
visual artifacts within a frame i.e it focuses on spatial inconsistencies. We can
enhance this model if we make it spatio-temporal so that it can spatio-temporal
inconsistencies across the frames. Capsule Net is an enhanced and more robust
version of CNN which will perform the feature extraction and detect the spatial
inconsistencies within the frame.
We can make it spatio-temporal by adding an LSTM to it that will detect the
temporal inconsistencies across the frames.
The question is how we can add an LSTM to the CapsNet model. We can add it
by removing the output capsules i.e, real and fake output capsules which are
doing the classification. In that case, there will be no routing algorithm runs
between the primary capsules and the output capsules as now there are no output
capsules.
In the CapsNet model, the input is passed through a VGG-19 pretrained model
which gives a feature vector and then it is passed through the primary capsules ,
each primary capsule will give a different feature vector. In the CapsNet model
we are using three primary capsules (light network), each capsule is giving an
output feature vector of 1x8, the output feature vector from all the primary
capsules are stacked together to form a single feature vector of size 3x8.
During the preprocessing stage , we capture 10 frames from each video of the
dataset. Now, these 10 frames are passed through this network , which will give
10 feature vectors of size 3x8. Now these 10 feature vectors are passed through a
many-to-one LSTM model as shown in the fig. to get the final result of the
classification.

![image3](https://user-images.githubusercontent.com/28837542/120933521-27742e00-c718-11eb-86b7-035814f011f8.png)


![image2](https://user-images.githubusercontent.com/28837542/120933525-28a55b00-c718-11eb-966c-34607cc97eee.png)



The LSTM used in the model is single layer and it consists of 256 hidden units.
The output produced by the last LSTM cell is passed through a fully connected layer
followed by a softmax which will give a classification score between 0 and 1 i.e, real
and fake.
To compute the loss, we used Cross - Entropy Loss Loss function and Adam Optimizer
is used to optimize the network with learning rate =0.0005 and β = 0.999.


### Results

From the analysis of models on both the datasets, CelebDF(v1) and FaceForensics++
dataset, the performance of the proposed Capsule + LSTM models
outperforms the CapsuleNet model proposed by Nguyen et al. by around 2%.
The accuracy acheived on Celeb-DF dataset is 95.8 and on FaceForensic++ dataset is 95.94.


This can be due to detection of spatio-temporal inconsistencies by the Capsule +LSTM
model which CapsuleNet is not able to perform as it is only detecting the visual artifacts
within a frame.
This shows that our proposed model outperforms the CapsuleNet model in detecting
deepfake.
