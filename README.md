# Project2
Project 2 for Udacity - Data scientist course

## Purpose
The purpose of this project is to use a provided dataset containing real messages that were sent during disaster events. the code provided in this GitHub will clean and process the data and then create a machine learning pipeline to categorize these events so that a message can be categorised and sent to an appropriate disaster relief agency.

This will also include a web app where an emergency worker can input a new message and get classification results in several categories

## Process_data.py

This code will clean and process the input dataset messages and categories. A copy of those dataset is provided in this GitHub

As inputs plese provide, in this order, message dataset filepath, categories dataset filepath and the database filepath (where the result of the cleaning process will be stored as an sql database)

If the code returns any error please make sure you insert sqlite:/// before the database filepath

How to run script (please adapt the below with your folder structure):  
python process_data.py messages.csv categories.csv sqlite:///Disaster.db


## Train_classifier.py

This code will built, train and evaluate a model to associate the right category to each message

As input please provide, in this order the database filepath (the same you have provided for Process_data.py, please remember if you have used sqlite:/// before the filepath) and the model filepath (which will terminate with the name of the model.pkl). The latter will be used to store the model as pickle

How to run script (please adapt the below with your folder structure):  
python train_classifier.py sqlite:///Disaster.db multimodel.pkl

## Run.py

This code is saved under the app folder together with the files go.html and master.html

This code will start a web app where an emergency worker can input a new message and get classification results in several categories. 
Once you run the code please follow the web link that will appear.

In order to run the code, you need to provide the database and the model you have created in the previous steps. Specifically, you need to provide the exact arguments you have included in train_classifier.py. If run.py code is saved in a different folder than train_classifier.py please make sure you provide the code with the correct folder path for the parameters.  

How to run script (please adapt the below with your folder structure):  
python run.py "sqlite:///../Disaster.db" "../multimodel.pkl"

