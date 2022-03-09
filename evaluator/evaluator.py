import random
import os, json

#import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import matplotlib.pyplot as plt

index_to_label_dict = {
    0: 'palm',
    1: 'finger_point',
    2: 'index_pick',
    3: 'middle_pick',
    4: 'victory',
    5: 'thumb_up',
    6: 'fist'
}

label_to_index_dict = {
    'palm': 0, 
    'finger_point': 1,
    'index_pick': 2,
    'middle_pick': 3,
    'victory': 4,
    'thumb_up': 5,
    'fist': 6
}


def get_expected_result(expected_result, json_data_input):
    #TODO: Make sure the path is correct to your setting.
    path_to_json = '../playground/data_collector/'
    for label in label_to_index_dict:
        label_path = path_to_json + label + '/labels/'
        if os.path.exists(label_path):
            for single_json_data in os.listdir(label_path):
                if single_json_data.endswith('.json'):
                    json_data_input.append(single_json_data)
                    expected_result.append(label_to_index_dict[label])



def eval_results(actual_result, json_data_inputs):
    dummyModel = dummy_model()
    for single_json_data in json_data_inputs:
        actual_result.append(dummyModel.predict()) 


class dummy_model():
    def __init__(self):
        self.num_labels = 6
    
    def predict(self):
        return random.randint(0,self.num_labels)

if __name__ == '__main__':
    expected_result = []
    json_data_inputs = []
    get_expected_result(expected_result, json_data_inputs)
    
    actual_result = []
    eval_results(actual_result, json_data_inputs)
    print(actual_result)

    #Precision Score = TP / (FP + TP)
    print('Precision: %.3f' % precision_score(expected_result, actual_result, average='micro'))

    #Recall Score = TP / (FN + TP)
    print('Recall: %.3f' % recall_score(expected_result, actual_result, average='micro'))

    #Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
    print('Accuracy: %.3f' % accuracy_score(expected_result, actual_result))

    #F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score/)
    print('F1 Score: %.3f' % f1_score(expected_result, actual_result, average='micro'))


    conf_matrix = confusion_matrix(y_true=expected_result, y_pred=actual_result)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.matshow(conf_matrix, cmap=plt.cm.Oranges, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
