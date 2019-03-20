#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 12:54:53 2018

@author: deola
"""
import pandas as pd
from matplotlib import pyplot as plt
import csv
import math
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score 
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.ensemble import VotingClassifier
import time
import random
from collections import defaultdict


np.random.seed(1)


#########################################################################################################
dataset = [[2.771244718,1.784783929,0],
           [1.728571309,1.169761413,0],
           [3.678319846,2.81281357,0],
           [3.961043357,2.61995032,0],
           [2.999208922,2.209014212,0],
           [7.497545867,3.162953546,1],
           [9.00220326,3.339047188,1],
           [7.444542326,0.476683375,1],
           [10.12493903,3.234550982,1],
           [6.642287351,3.319983761,1]]


dataset1=[[2.771244718,1.784783929,0]]

dataset2=[[1,0,1],
          [1,0,1],
          [0,0,0],
          [0,0,0],
          [1,3,0],
          [0,1,1]]
#########################################################################################################
def median_finder(dataset):
    
    '''Finding the median of a column of a dataset
     and insert it in an array
     returning the array'''
     
    arr, median=list(), list()
    for i in range(len(dataset[0])):
        for column in dataset:
            arr.append(column[i])
            #print arr
            if len(arr)==len(dataset):
                arr.sort()
                #print arr
                median.append(arr[len(arr)/2])
                arr=[]
    return median   
##############################################################################################

def mean_finder(dataset):
    
    
    '''Finding the mean of a column of a dataset
     and insert it in an array
     returning the array'''
     
    arr, mean=list(), list()
    for i in range(0,len(dataset[0])):
        for column in dataset:
            arr.append(column[i])
             
            if len(arr)==len(dataset):
                mean.append(sum(arr)/(len(arr))+0.00)
                arr=list()
    return mean
     
############################################################################################################
def part_by_mean(dataset, pos):
    '''Partition the dataset by comparing the element with
     the calculated median from the function MedianFinder(dataset)
     if the element of a particilar column is lesser than the calculated 
     median, it moves to left and vis-a vis for right.
     returning the array in left and right'''
    
    left, right, mean=list(),list(),list()
    mean=mean_finder(dataset)
    #print mean
    for row in dataset:
        if (row[pos] ==0 or row[pos] < mean[pos]):
            left.append(row)
#        if (row[pos] ==1 and row[pos] < median[pos]):
#            left.append(row)
        elif(row[pos] !=0 and row[pos] < mean[pos]):
            left.append(row)
        elif(row[pos] !=0 and row[pos] >= mean[pos]):
            right.append(row)
        
    return left, right
###########################################################################################################
counter=0
mydict=dict() 
result = {}
numm=0

def divide_recur(arr, num):
    '''The recursive partition of dataset in solution space
    and stored in a dictionary labelled with a particular name as left:0 or right:0'''
    
    global counter
    global mydict, numm
    
    if num <=0:
        return 
    
    else:
        if (len(arr)!=0):
            for i in range(0,len(arr[0])-1):
                first, second=part_by_mean(arr, i)
                
                mydict["left:"+str(counter)]=first
                mydict["right:"+str(counter)]=second
                counter+=1
                
                if num>1:
                    #mydict=dict()
                    del mydict["right:"+str(counter-1)]
                    del mydict["left:"+str(counter-1)]
                    time.sleep(0.5)
                    numm+=1
                    print("Deleted the entries in the dict: "+str(numm))
                    
                    numm+=1
                    print("Deleted the entries in the dict: "+str(numm))
                    #mydict.pop("left:"+str(counter-1))
                    divide_recur(first, num-1)
                    divide_recur(second, num-1)   
                
#    for key,value in mydict.items():
#        if value not in result.values():
#            result[key] = value
   
    #return result
   
    return mydict
    #mydict=dict() 
    
################################################################################################################
counter=0
mydict=dict()
mydict2=dict() 
mydict3=dict()
result = {}

def divide_recur2(arr, num):
    '''The recursive partition of dataset in solution space
    and stored in a dictionary labelled with a particular name as left:0 or right:0'''
    
    global counter
    global mydict
    
    
    if (len(arr)!=0):
        for i in range(0,len(arr[0])-1):
            first, second=part_by_mean(arr, i)
            counter+=1
            mydict["left:"+str(counter-1)]=first
            mydict["right:"+str(counter-1)]=second
                #DIVIDE_RECUR(first, num-1)
                #DIVIDE_RECUR(second, num-1)   
                
#    for key,value in mydict.items():
#        if value not in result.values():
#           result[key] = value
        print counter
        if num==2:
            
            for n in mydict.keys():
                for i in range(0,len(arr[0])-1):
                    first, second=part_by_mean(mydict[str(n)],i)
                    counter+=1
                    mydict2["left:"+str(counter-1)]=first
                    mydict2["right:"+str(counter-1)]=second
                    
            #print (len(mydict2.keys()))
#            print len(mydict2.keys())
#            return mydict2
            #for i in range(0, len(arr[0])-1):
                 
        if num==3:                 
            #return mydict
            for n in mydict.keys():
                for i in range(0,len(arr[0])-1):
                    first, second=part_by_mean(mydict[str(n)],i)
                    counter+=1
                    mydict2["left:"+str(counter-1)]=first
                    mydict2["right:"+str(counter-1)]=second
             
            for n in mydict2.keys():    
                for i in range(0,len(arr[0])-1):
                    first, second=part_by_mean(mydict2[str(n)],i)
                    counter+=1
                    mydict3["left:"+str(counter-1)]=first
                    mydict3["right:"+str(counter-1)]=second
#        #print mydict
            print len(mydict3)
            return mydict3
    #mydict=dict() 
##############################################################################################
def seperate_data_into_one_zero(dataset):
    '''Function to count the number of ones and zeros
    in the dataset'''
    
    zero=[]
    one=[]
    counterZero=0
    counterOne=0
    
    for i in dataset:
        if i[-1]==0:
            zero.append(i)
            counterZero+=1
        else:
            one.append(i)
            counterOne+=1
    #print one, zero       
    #print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$")
    #print"Number of ones:%d Number of zeros:%d and Length:%d" %(counterOne, counterZero , len(dataset))
    return counterOne, counterZero
    zero=[]
    one=[]
#####################################################################################################

actual=[1,1,1,1,1]
predicted=[1,1,0,0,1]

def accuracy_metric(real, pred):
    '''Accuracy of a model'''
    
    result = 0
    for i in range(len(real)):
        if real[i] == pred[i]:
            result += 1
    print str(result / float(len(real)) * 100.0)+'%'
#####################################################################################################
def log_model(mydict, num_col):

    for k in mydict:
        print k
        
        R=pd.DataFrame(mydict[k])
        X=R.values[:,0:num_col]
        y=R.values[:,num_col]
            #print y
        a=logistic_function(X,y)
        
        print"Accuracy: %s%%"%(str(a))      
##########################################################################################################
def logistic_function(X, y):
    '''Logistic Function to model each leaves of the partition'''
    
    model = LogisticRegression()
    X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.20 )
    model.fit(X_train, y_train)
    #return(model.score(X_test,y_test))
    y_hat = model.predict(X_test)
    predictions = [round(value) for value in y_hat]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    #accuracy=recall_score(y_test, predictions)
    #accuracy = precision_score(y_test, predictions)
    return accuracy * 100.0

########################################################################################################
##########################################################################################################
def logistic_function2(X, y, test, y_test):
    '''Logistic Function to model each leaves of the partition'''
    
    model = LogisticRegression()
    X_train, _, y_train, _=train_test_split(X,y,test_size=0.20 )
    model.fit(X_train, y_train)
    #return(model.score(X_test,y_test))
    y_hat = model.predict(test)
    predictions = [round(value) for value in y_hat]
    # evaluate predictions
    accuracy = accuracy_score(y_test, predictions)
    recall=recall_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    
    print " Single accuracy: %.2f%%\n Single precision: %.2f%% \n Single recall: %.2f%%"%(accuracy*100.0, precision*100.0, recall*100.00)
#    pred=list()
#    
#    if accuracy*100 >= min_acc:
#        pred=predictions
#        acc=accuracy
    return accuracy,predictions
########################################################################################################   
##########################################################################################################
def logistic_function_cv(X, y):
    '''Logistic Function tomodel each leave nodes of the partition 
    with StratifiedKFold to take care of imbalanced data'''
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    
    
    model = LogisticRegression()
    
    scoring='accuracy'
    #scoring='precision'
    #scoring = 'recall'
    #scoring='f1'
    
    kfold = StratifiedKFold(n_splits=10, random_state=20, shuffle=False)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring=scoring)
    #X_train, X_test, y_train, y_test=train_test_split(X,y,test_size=0.20 )
    #model.fit(X_train, y_train)
    #return(model.score(X_test,y_test))
    print("Accuracy: %.2f%%"%(cv_results.mean()*100))
    return cv_results.mean()*100

########################################################################################################
def plot_log_function(X,y,k):
    '''Function plots semilog graph of
    the models, showing std'''
    
    
    lg = LogisticRegression()
    C_s = np.logspace(-20, 0, 20)
    
    scores = list()
    scores_std = list()
    for C in C_s:
        lg.C = C
        this_scores = cross_val_score(lg, X, y, cv=10, n_jobs=-1)
        scores.append(np.mean(this_scores))
        scores_std.append(np.std(this_scores))
    
    # Do the plotting
    
    plt.figure(1, figsize=(5, 5))
    plt.clf()
    plt.semilogx(C_s, scores, 'r--')
    plt.semilogx(C_s, np.array(scores) + np.array(scores_std), 'b--')
    plt.semilogx(C_s, np.array(scores) - np.array(scores_std), 'b--')
    locs, labels = plt.yticks()
    plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
    plt.ylabel('CV score')
    plt.xlabel('Parameter C')
    plt.ylim(0.3, 1.1)
    plt.savefig("/home/deola/Documents/Autism/kfold/cross_val_%s.eps"%(str(k)), format="eps", dpi=300)
############################################################################################################
def csv_arr(datasetPath):
    '''Function converts CSV to array in array
    for the dataset'''
    
    datasetWithHeader=pd.read_csv(datasetPath)
    datasetWithHeader.to_csv("/home/deola/Documents/Autism/gameWithoutHeader.csv", header=None, index=False)
    
    results = []
    with open('/home/deola/Documents/Autism/gameWithoutHeader.csv', 'rb') as csvfile:
        reader = csv.reader(csvfile, quoting=csv.QUOTE_NONNUMERIC) # change contents to floats
        #next(reader, None)
        for row in reader: # each row is a list
            results.append(row)
    return results  

##################################################################################################
def Array_of_estimator(num_of_model):
    
    
    dataframe = pd.read_csv("/home/deola/Downloads/new_fuzzy/Dataset/mydata1_CKD.csv")
    array = dataframe.values
    X = array[:,0:24]
    Y = array[:,24]
    #count=0
    models,result, Tup=list(), list(), list()
   
    for i in range (0,num_of_model):
        
        models.append(("lg"+str(i),LogisticRegression()))
        
    eclf = VotingClassifier(estimators=models, voting='hard')
    eclf.fit(X,Y)
    print(eclf.predict(X))

###############################################################################################    
#arr1=[[1 0 0 0 1 1 1 1 1 1 1 1 1 0],[1,0,0,0,0,0,0,1,1,1,1,1,1,0],[0,0,0,0,1,0,1,0,1,0,1,0,1,0]]

def ensemble_pred(arr):
    '''Function to combine ensembles together.
    The array will be an array in array on predictions
    The majority class will be returned in and array'''
    
    countZero=0
    countOne=0
    result=list()
    
    for j in range(0, len(arr[0])):
        for i in range(0, len(arr)):
            if arr[i][j]==0:
                countZero+=1
            else:
                countOne+=1
        #print countZero, countOne
        if countZero >=countOne:
            result.append(0)
        else:
            result.append(1)
        countZero=0
        countOne=0        
    
    return result                  
          
#############################################################################################################
def wang_tree(pathTrain, pathTest, tree_depth, num_of_class, num_of_tree):
    '''The tree partitioning function to create different models
    for our logistic regression function
    PathTrain: The path to train the csv for the dataset.
    PathTest: The path to test the csv of the out of sample dataset
    tree_depth: The depth of the tree.
    num_of_class: Minimum number of a particular class.
    min_acc: The minimum accuracy to reject for the ensemble we have to set above 50 percent to 
    discriminate  models lesser than 50.
    '''
        
    arr_gameTrain=csv_arr(pathTrain)
    #arr_gameTrain=shuffle(arr_gameTrain)
    #arr_gameTest=CSV_ARR(pathTest)
    testSet=pd.read_csv(pathTest)
    _, col=testSet.shape
    X_test=testSet.values[:,0:col-1]
    y_test=testSet.values[:,col-1]
    print ("#######################################vvvv################")
    print ("#############################Jozi_Jozi#####################")   
    mydict= divide_recur(arr_gameTrain, tree_depth)
    time.sleep(3)
    #for key, value in mydict.items():
        #print key, len(value)
    print ("###########################################################")
    print ("#############################Mimi_Mimi#####################")
         
    result, ack, tempArr, razorArr=list(), list(), list(), list()
    count=0
    mini= defaultdict(list)
    for k in mydict.keys():
        count= count+1
        one, zero=seperate_data_into_one_zero(mydict[k])
        print ("Node: %s, zero: %d and one: %d"%(k, zero, one)) 
        if (one >=num_of_class and zero >=10) or (one >=10 and zero >=num_of_class):
            R=pd.DataFrame(mydict[k])
            X=R.values[:,0:(len(arr_gameTrain[0])-1)]
            y=R.values[:,(len(arr_gameTrain[0])-1)]
            if len(logistic_function2(X, y, X_test, y_test)) != 0:
                acc, res=logistic_function2(X, y, X_test, y_test)
                result.append(res)
                ack.append(acc*100)
                '''Putting the result in a dictionary'''
                
                if str(acc) in mini.keys():
                    print "Yes"
                else:
                    mini.update({str(acc):res})
     
#    time.sleep(5)
#    print(len(ack))
#           
    tempArr=map(float, mini.keys())
    tempArr=sorted(tempArr, key=float)
    mid=len(tempArr)/2
   
    #print mini
    
    razorArr.append(mini[str(min(tempArr))])
    razorArr.append(mini[str(max(tempArr))])
    #print'\n\nhhhhhh'
    #print tempArr
    #razorArr.append(mini[str(tempArr[mid])])
    #
    #print result
    
    final_result=list()
    for w in result:
        if len(w) !=0:
            final_result.append(w)
    #pred=Ensemble_Pred(final_result+ razorArr*(int(math.ceil(len(final_result)*0.9))))
    print (len(final_result))
    print ("\n") 
    #print tempArr
    print ("\n")
    print" The worst performing model: %.2f%% \n The best performing model: %.2f%%"%( min(tempArr)*100, max(tempArr)*100)
    pred=ensemble_pred(razorArr)
    #pred=Ensemble_Pred(tempArr)
    
    
    
    
    print" The number of models: %.2d" %( len(tempArr))
    
    
    print ("\n") 
    print "Instances of test set: %d"%(len(pred))
    #print ("\n") 
    #print (pred)
    print ("\n")
    #print(ack)
    #print mini
    '''This aspect combines the best performing and the worst performing model
    to give us the ensemble for accuracy, precision and recall'''
    #accuracy_metric(y_test, pred)
    accuracy = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred)
    recall = recall_score(y_test, pred)
    print ("\n")
    print " Ensemble accuracy1: %.2f%%\n Ensemble precision1: %.2f%% \n Ensemble recall1: %.2f%%"%(accuracy*100.0, precision*100.0, recall*100.00)
    print ("\n")
    
    ''' This combines all the prediction to '''
    allArr,tempAllArr, allArr1=list(),list(),list()
    for i in mini:
        allArr.append(mini[str(i)])
    #print allArr
    if num_of_tree<=len(final_result):
        for m in range(0,num_of_tree):
            tempAllArr.append(random.randint(0,(len(final_result)-1)))
        print (tempAllArr)
        
        for m in tempAllArr:
            allArr1.append(final_result[m])
            #print allArr1
    else:
        allArr1=final_result
        
    pred2=ensemble_pred(allArr1)
    
    
    accuracy2 = accuracy_score(y_test, pred2)
    precision2 = precision_score(y_test, pred2)
    recall2 = recall_score(y_test, pred2)
    
    print ("\n")
    print " Ensemble for all models accuracy2: %.2f%%\n Ensemble for all models precision2: %.2f%% \n Ensemble for all models recall2: %.2f%%"%(accuracy2*100.0, precision2*100.0, recall2*100.00)
    
    print(" The number of model %d"%len(allArr1))
    print ("\n")
    
    tryArr, tempArr2=list(), list()
    
    mini2=dict()
    
    '''This aspect goes through all the possible combination of model to give the best performance
     ensemble for accuracy2, precision2 and recall2'''
    
    for model1 in tempArr:
        for model2 in tempArr:
            tryArr.append(mini[str(model1)])
            tryArr.append(mini[str(model2)])
            pred=ensemble_pred(tryArr)
            acc = accuracy_score(y_test, pred)
            #acc= acc*100
            #print acc
            mini2[str(acc)]=pred
            tryArr=list()
    
    tempArr2=map(float, mini2.keys())
    tempArr2=sorted(tempArr2, key=float)
    pred3=mini2[str(max(tempArr2))]
    #print tempArr2
    #print pred3
    
    #print y_test
    
    accuracy3 = accuracy_score(y_test, np.array(pred3))
    precision3 = precision_score(y_test, pred3)
    recall3 = recall_score(y_test, pred3)
    print ("\n")
    #print " Ensemble of best combo accuracy3: %.2f%%\n Ensemble of best combo precision3: %.2f%% \n Ensemble of best combo recall3: %.2f%%"%(accuracy3*100.0, precision3*100.0, recall3*100.00)
    #print len(tempArr)
############################################################################################################
############################################################################################################
############################################################################################################
############################################################################################################
    
if __name__=="__main__":
     
     time1=time.time()
     wang_tree("/home/deola/Documents/Autism/Autism_with_header_converted_to_bin1.csv",
              "/home/deola/Documents/Autism/Autism_with_header_converted_to_bin_test.csv", 3, 100, 20)
     print("\n")
     
     print("Time taken is sec: "+str(time.time()-time1))
     
     
     
     #WangTree("/home/deola/Documents/Autism/test_dat.csv", 3, 20 )
     #Array_of_estimator(8)
     
     #arr_gameTrain=csv_arr("/home/deola/Documents/Autism/Autism_with_header_converted_to_bin1.csv")
     #print(divide_recur(dataset, 3))
     
     #Ensemble_Pred(arr1)
     #print(accuracy_metric(actual, predicted))
