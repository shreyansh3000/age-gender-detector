# Gender and Age Detection Python Project

First introducing you with the terminologies used in this advanced python project of gender and age detection –

## What is Computer Vision?

Computer Vision is the field of study that enables computers to see and identify digital images and videos as a human would. The challenges it faces largely follow from the limited understanding of biological vision. Computer Vision involves acquiring, processing, analyzing, and understanding digital images to extract high-dimensional data from the real world in order to generate symbolic or numerical information which can then be used to make decisions. The process often includes practices like object recognition, video tracking, motion estimation, and image restoration.

## What is OpenCV?

OpenCV is short for Open Source Computer Vision. Intuitively by the name, it is an open-source Computer Vision and Machine Learning library. This library is capable of processing real-time image and video while also boasting analytical capabilities. It supports the Deep Learning frameworks TensorFlow, Caffe, and PyTorch.

## What is a CNN?

<p>A <em><strong><a href="https://data-flair.training/blogs/convolutional-neural-networks/">Convolutional Neural Network</a></strong></em> is a deep neural network (DNN) widely used for the purposes of image recognition and processing and <em><strong><a href="https://data-flair.training/blogs/nlp-natural-language-processing/">NLP</a></strong></em>. Also known as a ConvNet, a CNN has input and output layers, and multiple hidden layers, many of which are convolutional. In a way, CNNs are regularized multilayer perceptrons.</p>

# Gender and Age Detection Python Project- Objective

<p>To build a gender and age detector that can approximately guess the gender and age of the person (face) in a picture using <a href="https://data-flair.training/blogs/deep-learning/"><em><strong>Deep Learning</strong></em></a> on the Adience dataset.</p>

# Gender and Age Detection – About the Project

<p>In this Python Project, we will use Deep Learning to accurately identify the gender and age of a person from a single image of a face. We will use the models trained by <a href="https://talhassner.github.io/home/projects/Adience/Adience-data.html" rel="nofollow" onclick="javascript:window.open('https://talhassner.github.io/home/projects/Adience/Adience-data.html'); return false;">Tal Hassner and Gil Levi</a>. The predicted gender may be one of ‘Male’ and ‘Female’, and the predicted age may be one of the following ranges- (0 – 2), (4 – 6), (8 – 12), (15 – 20), (25 – 32), (38 – 43), (48 – 53), (60 – 100) (8 nodes in the final softmax layer). It is very difficult to accurately guess an exact age from a single image because of factors like makeup, lighting, obstructions, and facial expressions. And so, we make this a classification problem instead of making it one of regression.</p>

## The CNN Architecture

The convolutional neural network for this python project has 3 convolutional layers:

<ul><li>Convolutional layer; 96 nodes, kernel size 7</li><li>Convolutional layer; 256 nodes, kernel size 5</li><li>Convolutional layer; 384 nodes, kernel size 3</li></ul>

It has 2 fully connected layers, each with 512 nodes, and a final output layer of softmax type.

To go about the python project, we’ll:

<ul><li>Detect faces</li><li>Classify into Male/Female</li><li>Classify into one of the 8 age ranges</li><li>Put the results on the image and display it</li></ul>

# The Dataset

The dataset has been linked in the main.py program......when u run the program, dataset will be downloaded automatically and load to the program training set. If u want use a different dataset. You can through the below link from Kaggle datasets.

<p>For this python project, we’ll use the Adience dataset; the dataset is available in the public domain and you can find it <em><strong><a href="https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification" onclick="javascript:window.open('https://www.kaggle.com/ttungl/adience-benchmark-gender-and-age-classification'); return false;">here</a></strong></em>. This dataset serves as a benchmark for face photos and is inclusive of various real-world imaging conditions like noise, lighting, pose, and appearance. The images have been collected from Flickr albums and distributed under the Creative Commons (CC) license. It has a total of 26,580 photos of 2,284 subjects in eight age ranges (as mentioned above) and is about 1GB in size. The models we will use have been trained on this dataset.</p>

## Prerequisites

You’ll need to install OpenCV (cv2) to be able to run this project. You can do this with pip-

   * $ pip install opencv-python
   
Other packages you’ll be needing are math and argparse, but those come as part of the standard Python library.

# Steps for practicing gender and age detection python project

<p>1. <a href="https://drive.google.com/file/d/1yy_poZSFAPKi0y2e2yj9XDe1N8xXYuKB/view" onclick="javascript:window.open('https://drive.google.com/file/d/1yy_poZSFAPKi0y2e2yj9XDe1N8xXYuKB/view'); return false;"><strong>Download this zip</strong></a>. Unzip it and put its contents in a directory you’ll call gad.</p>

The contents of this zip are:

<ul><li>opencv_face_detector.pbtxt</li><li>opencv_face_detector_uint8.pb</li><li>age_deploy.prototxt</li><li>age_net.caffemodel</li><li>gender_deploy.prototxt</li><li>gender_net.caffemodel</li><li>a few pictures to try the project on</li></ul>

<p>For face detection, we have a .pb file- this is a protobuf file (protocol buffer); it holds the graph definition and the trained weights of the model. We can use this to run the trained model. And while a .pb file holds the protobuf in binary format, one with the .pbtxt extension holds it in text format. These are TensorFlow files. For age and gender, the .prototxt files describe the network configuration and the .caffemodel file defines the internal states of the parameters of the layers.</p>

<p>2. We use the argparse library to create an argument parser so we can get the image argument from the command prompt. We make it parse the argument holding the path to the image to classify gender and age for.</p>

3. For face, age, and gender, initialize protocol buffer and model.

4. Initialize the mean values for the model and the lists of age ranges and genders to classify from.

5. Now, use the readNet() method to load the networks. The first parameter holds trained weights and the second carries network configuration.

6. Let’s capture video stream in case you’d like to classify on a webcam’s stream. Set padding to 20.

7. Now until any key is pressed, we read the stream and store the content into the names hasFrame and frame. If it isn’t a video, it must wait, and so we call up waitKey() from cv2, then break.

8. Let’s make a call to the highlightFace() function with the faceNet and frame parameters, and what this returns, we will store in the names resultImg and faceBoxes. And if we got 0 faceBoxes, it means there was no face to detect.
Here, net is faceNet- this model is the DNN Face Detector and holds only about 2.7MB on disk.

<ul><li>Create a shallow copy of frame and get its height and width.</li><li>Create a blob from the shallow copy.</li><li>Set the input and make a forward pass to the network.</li><li>faceBoxes is an empty list now. for each value in 0 to 127, define the confidence (between 0 and 1). Wherever we find the confidence greater than the confidence threshold, which is 0.7, we get the x1, y1, x2, and y2 coordinates and append a list of those to faceBoxes.</li><li>Then, we put up rectangles on the image for each such list of coordinates and return two things: the shallow copy and the list of faceBoxes.</li></ul>

9. But if there are indeed faceBoxes, for each of those, we define the face, create a 4-dimensional blob from the image. In doing this, we scale it, resize it, and pass in the mean values.

10. We feed the input and give the network a forward pass to get the confidence of the two class. Whichever is higher, that is the gender of the person in the picture.

11. Then, we do the same thing for age.

12. We’ll add the gender and age texts to the resulting image and display it with imshow().

## Python Project Examples for Gender and Age Detection

Let’s try this gender and age classifier out on some of our own images now.

<p><strong>Output:</strong></p>

<p><a href="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/09/python-project-example-output-1.png" onclick="javascript:window.open('https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/09/python-project-example-output-1.png'); return false;" class=""><img class="aligncenter wp-image-70000 tc-smart-load-skip tc-smart-loaded" src="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/09/python-project-example-output-1.png" alt="python open source project" width="206" height="313" sizes="(max-width: 206px) 100vw, 206px" srcset="https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/09/python-project-example-output-1.png 478w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/09/python-project-example-output-1-99x150.png 99w, https://d2h0cx97tjks2p.cloudfront.net/blogs/wp-content/uploads/sites/2/2019/09/python-project-example-output-1-198x300.png 198w" style="display: block;"></a></p>

# Summary

In this python project, we implemented a CNN to detect gender and age at real time using web cam.

If you enjoyed the above python project, do comment and let us know your thoughts.

Happy learning😊

Follow Me On Instagram at <a href = "https://www.instagram.com/_hemanth_shetty__/">@_hemanth_shetty__</a>

#### ThankYou!
