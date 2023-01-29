
# Flower Recognition

The flower recognition system based on image processing has been developed. This system uses edge and color characteristics of flower images to classify flowers, and then at the end of all operations, the flower type is predicted.


## CNN

Convolutional Neural Network is a Deep Learning algorithm specially designed for working with Images and videos. It takes images as inputs, extracts and learns the features of the image, and classifies them based on the learned features.
## Description

- Total images are 4317
- various graphical approach to visualise the images.
- Dataset containing input images of various types of flowersw
- Usage of deep learning methods and torchvision library.

## Usage and Installation

```

import numpy as np
import pandas as pd
import plotly as plot
import plotly.express as px
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import sklearn 
import cufflinks as cf
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import plotly.offline as pyo
from plotly.offline import init_notebook_mode, plot, iplot
import opendatasets as od
import os
import shutil
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision.utils import make_grid



Sequential model
import keras
from keras.models import Sequential
from keras.layers import Dense





```


## Acknowledgements

  ['https://www.kaggle.com/alxmamaev/flowers-recognition'](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)



## Appendix

A very crucial project in the realm of data science and image processing and deep learning. Multiple high level concepts were used.


## Running Tests

To run tests, run the following command

```bash
  npm run test
```

We got an accuracy of 79.8 % in this model. We ran various epochs and used efficiet data cleansing techniques to get to this.

## Used By

The project is used by a lot of social media companies to analyse their market.

## OUTPUT
![image](https://user-images.githubusercontent.com/92213377/215313443-4cde0816-49ed-4016-ad83-ee051710c5c2.png)
![image](https://user-images.githubusercontent.com/92213377/215313457-22254436-b15a-42e0-b29e-fe6b213cfc9f.png)

