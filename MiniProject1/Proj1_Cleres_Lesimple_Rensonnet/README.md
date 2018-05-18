# DeepLearning2018

### Requierements 
Hardware
For Mac OSX - MacBook Pro (Retina, 15-inch, Mid 2014)
CPU: 2.5 GHz Intel Core i7 GPU: NVIDIA GeForce GT 750M 2 GB

anaconda

conda install pytorch torchvision -c pytorch
pip install -U scikit-learn
scipy
matplotlib
numpy

Please find in the folder the test.py file which only needs to be launched with the command python test.py. 

This piece of code runs the model number 3 that we implemented. The code runs the trainning and testing for the best performing CNN that was implemented by our group (David Cleres, Nicolas Lesimple & Gaëtan Rensonnet). 
There are no paramters to give to the programm. During runtime the user is informed about the loss at each epoch and the accuracy of the model on the validation and the training datasets. 

The utility file cointained all the functions that were useful all along the project in order to perform the preprocessing (adding white noise, cutting the last part of the signal because it was very noisy, computing the FFT, ...). 
The model file cointained a summary of all the different models that we tried all along of the project. 

The dlc_bci file was given and left unchanged. 
