# oxford-flowers-image-classifier

![Contributors](https://img.shields.io/github/contributors/walidsi/oxford-flowers-image-classifier?style=plastic)
![Forks](https://img.shields.io/github/forks/walidsi/oxford-flowers-image-classifier)
![Downloads](https://img.shields.io/github/downloads/walidsi/oxford-flowers-image-classifier/total)
![Stars](https://img.shields.io/github/stars/walidsi/oxford-flowers-image-classifier)
![Licence](https://img.shields.io/github/license/walidsi/oxford-flowers-image-classifier)
![Issues](https://img.shields.io/github/issues/walidsi/oxford-flowers-image-classifier)

### Goal
The goal of the project is to train an image classifier to recognize different species of flowers.

### Process
- Load the oxford flowers image dataset and create a pipeline.
- Build and Train an image classifier on this dataset using transfer learning with mobilenet_v2 as the base pre-trained network from TensorFlow Hub.
- Save and use the trained model to perform inference on unseen flower images.

### Results
- After training the model for 5 epochs, we achieved a training accuracy of 98.63% and an accuracy of 74.695% on the testing set.
- The model was saved and later used in a command line script and a Flask web application to predict the species of flowers from random images. The web app can be found at https://oxford-flowers.azurewebsites.net

### Install

This project requires **Python 3.x** and the following Python libraries installed:

- [NumPy](http://www.numpy.org/)
- [Pandas](http://pandas.pydata.org)
- [matplotlib](http://matplotlib.org/)
- [Tensorflow 2.9](https://tensorflow.org/)

You will also need to have software installed to run and execute an [iPython Notebook](http://ipython.org/notebook.html)

### Code

Code is provided in the `Project_Image_Classifier_Project.ipynb` notebook file.

### Run

In a terminal or command window, navigate to the top-level project directory (that contains this README) and run one of the following commands:

```bash
ipython notebook Project_Image_Classifier_Project.ipynb
```  
or
```bash
jupyter notebook Project_Image_Classifier_Project.ipynb
```

This will open the iPython Notebook software and project file in your browser.

### Data

The dataset consists of the following images downloaded using tensorflow_datasets module:
  - https://www.tensorflow.org/datasets/catalog/oxford_flowers102

**Features**
- Features are defined as the colors of the pixels in the imae(s).
