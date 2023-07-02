# Azure Machine Learning

## Description

Doctors use chest xrays to determine whether someone may have a disease or not. A few diseases identified partly or entirely from chest xrays are Pneumonia and COVID-19. Sometimes the difference between a healthy xray and an xray indicating illness is very small. This means doctors need to be very well trained to identify these small differences. Lots of things can go wrong, such as a doctor missing a small detail after working a 12 hour shift. Since this can be a matter of life and death in extreme cases, we would like to avoid these cases as much as possible. This is why this project aims to create a chest xray classifier using Azure Machine Learning Studio.

## Data

The dataset consits of 317 chest X-rays and can be found on [Kaggle](https://www.kaggle.com/). Each of the images is associated with a classification: `Normal`, `Viral Pneumonia`, or `COVID-19`. These images are all black and white and have varying dimensions. The folder structure of the data is as follows:

```
- Train
 |--- Covid
 |--- Normal
 |--- Viral Pneumonia
- Test
 |--- Covid
 |--- Normal
 |--- Viral Pneumonia
```

## Run

'azure-model-training.ipynb' follows the getting started example on Azure Machine Learning (under the notebooks section) and provides all the code necessary for training the model. This notebook is very well documented so I highly recommend taking a look if you want to learn more.

### On Azure

To run this in the cloud, simply open up `azure-model-training.ipynb` and run each cell. In the first cell, you will need to enter your own credentials such as subscription id, as I am not providing mine to the public. This will create the azure job and execute it for you. a pretrained ResNet18 model is used in this case to limit the size and resources needed for this POC project.

### Locally

if you would like to run locally, first install the necessary requirements with `pip install -r requirements.txt`

Then run the following command: `python src\main.py --data chest_xrays`

