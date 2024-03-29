# abstract-art-classification

This is my final project for the Fall 2019 offering of the course CS230: Deep Learning at Stanford University.

CNNs generally perform well on many image classification tasks that require distinguishing between the objects represented by images. However, I hypothesize that the problem of distinguishing between professional amateur abstract artwork cannot be so easily solvable by standard CNN architectures. Therefore, in this project I address the problem of distinguishing between professional and amateur abstract artwork using different CNN architectures and deep learning techniques.

## Data

To download the DeviantArt and MART datasets, please contact me at sharmant@stanford.edu. Once you have downloaded the datasets, run

```python3 preprocess.py```
  
to preprocess, augment, and pickle the train, dev, and test datasets.

## Models

The Baseline model is defined in `baseline.py`, and the Shallow model is defined in `shallow.py`.

### Baseline Model

The baseline model is a CNN with the following architecture:

- Input
- CONV1 (f=3, p=1, a=1) * 8
- ReLU + MaxPool2D (2,2)
- CONV2 (f=3, p=1, a=1) * 16
- ReLU + MaxPool2D (2,2)
- CONV3 (f=3, p=1, a=1) * 32
- ReLU + MaxPool2D (2,2)
- CONV4 (f=3, p=1, a=1) * 64
- ReLU + MaxPool2D (2,2)
- Dropout
- FC1 (256)
- ReLU
- FC2 (84)
- ReLU
- Dropout
- Softmax (2)

### Shallow Model

The shallow model is a "shallow" version of the baseline model in that it has shallower channel depths compared to the CNN in the baseline model. 
The shallow model has the following architecture:

- Input
- CONV1 (f=3, p=1, a=1) * 8
- ReLU + MaxPool2D (2,2)
- CONV2 (f=3, p=1, a=1) * 8
- ReLU + MaxPool2D (2,2)
- CONV3 (f=3, p=1, a=1) * 8
- ReLU + MaxPool2D (2,2)
- CONV4 (f=3, p=1, a=1) * 8
- ReLU + MaxPool2D (2,2)
- Dropout
- FC1 (256)
- ReLU
- FC2 (84)
- ReLU
- Dropout
- Softmax (2)

## Running Experiments

To run the baseline with all default arguments, run

```python3 baseline.py```

To run the shallow model with all default arguments, run

```python3 shallow.py```
  
Examples of running the baseline and shallow models with custom parameters for learning rate and dropout probability are as follows:

```python baseline.py --lr 0.001```
  
```python shallow.py --dropout 0.4```
