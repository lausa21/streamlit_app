import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
import warnings
from sklearn.metrics import classification_report, accuracy_score
import pickle

class DataHandler: 
    def __init__(self, file_path):
        self.file_path = file_path
        self.data, self.input_df, self.output_df = [None] * 3

    def load_data(self):
        self.data = pd.read_csv(self.file_path)
        
    def create_input_output(self, target_column):
        self.input_df = self.data.drop(target_column, axis=1)
        self.output_df = self.data[target_column]

class ModelHandler: 
    def __init__(self, input_data, output_data):
        self.input_data = input_data
        self.output_data = output_data
        self.x_train, self.x_test, self.y_train, self.y_test, self.y_predict = [None] * 5
    
    def del_id(self, id_column):
        self.input_data = self.input_data.drop(id_column, axis=1)

    def checkMissingValue(self):
        print("Missing values in input data:\n", self.input_data.isna().sum())
        print("\nMissing values in output data:\n", self.output_data.isna().sum())


    def split_data(self, test_size=0.2, random_state=42):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
        self.input_data, self.output_data, test_size=test_size, random_state=random_state)
    
    def encoding(self):
        self.label_enc = LabelEncoder()
        self.y_train = self.label_enc.fit_transform(self.y_train)
        self.y_test2 = self.label_enc.transform(self.y_test)

    def createModel(self):
        self.model = RandomForestClassifier()

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)

    def makePrediction(self):
        self.y_predict = self.model.predict(self.x_test) 
    
    def inv_transform(self):
        self.y_predict2 = self.label_enc.inverse_transform(self.y_predict)
        
    def createReport(self):
        print('\nClassification Report\n')
        print(classification_report(self.y_test, self.y_predict2, target_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']))

    def evaluate_model(self):
        return accuracy_score(self.y_test, self.y_predict2)

    def save_model_to_file(self, filename, encoder_filename):
        with open(filename, 'wb') as file:  
            pickle.dump(self.model, file)
        with open(encoder_filename, 'wb') as file:
            pickle.dump(self.label_enc, file) 


file_path = 'Iris.csv'  
data_handler = DataHandler(file_path)
data_handler.load_data()
data_handler.create_input_output('Species')
input_df = data_handler.input_df
output_df = data_handler.output_df

model_handler = ModelHandler(input_df, output_df)
model_handler.del_id('Id')
model_handler.checkMissingValue()
model_handler.split_data()
model_handler.encoding()
model_handler.createModel()
model_handler.train_model()
model_handler.makePrediction()
model_handler.inv_transform()
model_handler.createReport()
print("Model Accuracy:", model_handler.evaluate_model())
model_handler.save_model_to_file('trained_model.pkl', 'label_encoder.pkl')
