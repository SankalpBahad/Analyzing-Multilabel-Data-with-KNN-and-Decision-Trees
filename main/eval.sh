#!/bin/bash

if [ $# -ne 1 ]; then
    echo "Enter test data file"
    exit 1
fi

dataset_file="$1"

if [ ! -e "$dataset_file" ]; then
    echo "No such file exists! "
    exit 1
fi

python3 - << END_PYTHON_CODE 

import numpy as np
data=np.load("data.npy",allow_pickle=True)

train_data=[]
test_data=[]
validation_data=[]

# we will split the data in train:validation in 80:20 respectively

train_data=data[:1200]
validation_data=data[1200:]

# test_data=np.load("$dataset_file",allow_pickle=True)
test_data=validation_data

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class KNN():

    def __init__(self, encoder_type, k, distance_metric):
        self.encoding=encoder_type
        self.dist_metric=distance_metric
        self.k=k

    def get_output(self, entry, train_data):
        result=self.find_distance(entry,train_data)
        # print(result)
        result_k=self.get_top_k(result)
        # print(result_k)
        prediction=self.get_prediction(result_k)
        return prediction
    
    def find_distance(self, entry, train_data):
        result=[]
        if(self.dist_metric == 0):
            # Manhattan
            sum=0
            for train in train_data:
                res=np.abs(entry[self.encoding+1][0]-train[self.encoding+1][0])
                sum=np.sum(res)
                result.append([sum,train[3]])
            
        if(self.dist_metric == 1):
            # Euclidean
            sum=0
            for train in train_data:
                res=np.abs(entry[self.encoding+1][0]-train[self.encoding+1][0])
                res=np.square(res)
                sum=np.sum(res)
                sum=np.sqrt(sum)
                result.append([sum,train[3]])
            # return sum
        if(self.dist_metric == 2):
            # Cosine distance
            sum=0
            for train in train_data:
                res=np.dot(entry[self.encoding+1][0],train[self.encoding+1][0])
                mod_train=np.sqrt(np.sum(np.square(train[self.encoding+1][0])))
                mod_test=np.sqrt(np.sum(np.square(entry[self.encoding+1][0])))
                sum=res/(mod_train*mod_test)
                sum=1-sum
                result.append([sum,train[3]])
        # print(result)
        return result

    def get_top_k(self,result):
        result.sort()
        result=result[:self.k]
        # print(result)
        return result
        
    def get_prediction(self,result):
        label_counts={}
        for i in result:
            if i[1] not in label_counts.keys():
                label_counts[i[1]]=1
            else:
                label_counts[i[1]]+=1
            #     print(label_counts,end="\t")
            #     print(entry[3])
        max_label=list(label_counts.keys())[0]
        max_ct=label_counts[list(label_counts.keys())[0]]
        for i in label_counts.keys():
            if label_counts[i]>max_ct:
                max_ct=label_counts[i]
                max_label=i
        max_labels=[]
        for i in label_counts.keys():
            if(i==max_label):
                max_labels.append(i)
        prediction=""
        for i in result:
            if i[1] in max_labels:
                prediction=i[1]
                break 
        # print(prediction)
        return prediction

    def get_metrics(self,y_pred,y_true):
        f1=f1_score(y_true,y_pred, average="weighted", zero_division=1)
        rec=recall_score(y_true,y_pred, average="weighted", zero_division=1)
        prec=precision_score(y_true,y_pred, average="weighted", zero_division=1)
        acc=accuracy_score(y_true,y_pred)
        return [f1,rec,prec,acc] 
        
y_true=[]
y_pred=[]
output=[]
for enc in range(2):
    for d in range(3):
        for k in range(1,3):
            y_true = []
            y_pred = []
            total = 0
            for entry in validation_data:
                knn = KNN(enc, k, d)
                prediction = knn.get_output(entry, train_data)
                y_pred.append(prediction)
                y_true.append(entry[3])
            f1,recall,precision,accuracy=knn.get_metrics(y_true,y_pred)
            output.append([accuracy,[k,enc,d,f1,recall,precision]])

output.sort(reverse=True)
best_resnet=[]
best_vit=[]
for i in output:
    if i[1][1]==0:
        best_resnet=i
        break
for i in output:
    if i[1][1]==1:
        best_vit=i
        break

print("","-"*124)
print("|  ","Encoding".ljust(20),"|  ","F1".ljust(20),"|  ", "Precision".ljust(20),"|  ", "Recall".ljust(20),"|  ", "Accuracy".ljust(20),"|")
print("","-"*124)

k=best_resnet[1][0]
enc=best_resnet[1][1]
d=best_resnet[1][2]
y_pred=[]
y_true=[]
for entry in test_data:
    knn = KNN(enc, k, d)
    prediction = knn.get_output(entry, train_data)
    y_pred.append(prediction)
    y_true.append(entry[3])
f1,recall,precision,accuracy=knn.get_metrics(y_true,y_pred)
encoding="ResNet"
print("|  ",encoding,(19-len(encoding))*' ',"|  ",f"{f1:.12f}".ljust(20),"|  ", f"{precision:.12f}".ljust(20),"|  ", f"{recall:.12f}".ljust(20),"|  ", f"{accuracy:.12f}".ljust(20),"|")
print("","-"*124)

k=best_vit[1][0]
enc=best_vit[1][1]
d=best_vit[1][2]
y_pred=[]
y_true=[]
for entry in test_data:
    knn = KNN(enc, k, d)
    prediction = knn.get_output(entry, train_data)
    y_pred.append(prediction)
    y_true.append(entry[3])
f1,recall,precision,accuracy=knn.get_metrics(y_true,y_pred)
encoding="VIT"
print("|  ",encoding,(19-len(encoding))*' ',"|  ",f"{f1:.12f}".ljust(20),"|  ", f"{precision:.12f}".ljust(20),"|  ", f"{recall:.12f}".ljust(20),"|  ", f"{accuracy:.12f}".ljust(20),"|")
print("","-"*124)

END_PYTHON_CODE
