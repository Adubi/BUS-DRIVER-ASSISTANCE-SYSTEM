#  AUTONOMOUS PUBLIC BUS DRIVER ASSISTANCE SYSTEM

The aim of this project is to develop an assistance system for drivers in autonomous bus public transportation. Deep learning techniques are employed to create models for object detection (CNN), distance estimation (DNN), and speed estimation (CNN+ OPTICAL FLOW) using data from a monocular camera. The output of these models, which includes object information, distance, and speed, is then utilized as input for predicting the speed of the bus using an Artificial Neural Network (DNN) model.

![1st proj](https://github.com/Adubi/BUS-DRIVER-ASSISTANCE-SYSTEM/assets/44330438/fb7149b3-f597-4981-97f7-301ec8623476)


## OBJECT DETECTION
Object detection was used to make our car able to see like the driver. YOLO is recognized for its speed and accuracy, with it being a single-stage object detector architecture that is composed of three components. For YOLOv5 models it used CSP-Darknet53 as a backbone, SPP and PANet in the model neck and the head. This project uses YOLOv5s model (fast, small).




https://github.com/Adubi/BUS-DRIVER-ASSISTANCE-SYSTEM/assets/44330438/176355b6-edf4-44d7-8ea0-4cd06c49c1b9




## DISTANCE ESTIMATION
A deep learning model was used to estimate the distance between an object and the camera by using the distance estimation model (DNN) - trained on the KITTI dataset - to predict the distance from the results of object detection. Created a Deep neural network model with 5 layers, using ReLu activation. Defined loss function using mean square error, and Adam optimizer.


![image](https://github.com/Adubi/BUS-DRIVER-ASSISTANCE-SYSTEM/assets/44330438/61741bde-932c-4eef-93b2-fc31b48b8050)

## SPEED ESTIMATION
A deep learning model was used to estimate the distance between an object and the camera by using the distance estimation model (DNN) - trained on the KITTI dataset - to predict the distance from the results of object detection. Created a Deep neural network model with 5 layers, using ReLu activation. Defined loss function using mean square error, and Adam optimizeR.

The results of all the previous steps is stored in a CSV file which then uses all the features (object, bounding box, and distance) as input for the cars speed prediction by using DNN (Deep Neural Network) model. Created the model with 5 layers, using ReLu activation, kernel initializer as ‘normal’, and Batch Normalization. Defined Adam optimizer, and loss function using mean square logarithms error.



#  PROJECT POSTER

![image](https://github.com/Adubi/BUS-DRIVER-ASSISTANCE-SYSTEM/assets/44330438/8f75dd6d-de20-46a7-9139-eb7b9665a353)

# Acknowledgement
The code from this project was modified from:

[KITTI Distance estimation](https://github.com/harshilpatel312/KITTI-distance-estimation)

[Vehicle Distance Estimation](https://github.com/RmdanJr/vehicle-distance-estimation)

[Speed Estimation of car with Optical Flow](https://github.com/laavanyebahl/speed-estimation-of-car-with-optical-flow)
