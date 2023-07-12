import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import random
import warnings
import matplotlib.pyplot as plt

# Filter a specific warning by category
warnings.filterwarnings("ignore")

# concatenated_data = pd.DataFrame()
# csv_files = ['time_1_16.csv','time_16_21.csv','time_22_27.csv']
# data_frames = []
# for file in csv_files:
#     # Read each CSV file
#     data = pd.read_csv(file)
#     data_frames.append(data)
#     # Append the data to the concatenated DataFrame
# concatenated_data = pd.concat(data_frames, ignore_index=True)

# # Save the concatenated data to a new CSV file
# concatenated_data.to_csv('times.csv', index=False)

# Assuming you have your dataset stored in a pandas DataFrame called "data"
# and the target variable in a column named "target"

# Splitting the dataset into features and target
arr = ["ochiais", "jaccards", "gp13s", 
       "wong1s", "wong2s", "wong3s", "tarantulas", 
       "amples", "RussellRaos", "SorensenDices", "Kulczynski1s", 
       "SimpleMatching", "M1", "RogersTanimoto", "Hamming", "Ochiai2", 
       "Hamann", "dice", "Kulczynski2", "Sokal", "M2", "Goodman", "Euclid", "Anderberg", 
       "Zoltar", "ER1a", "ER1b", "ER5a", "ER5b", "ER5c", "gp02", "gp03", "gp19","col"]

data = pd.read_csv('times.csv')

data['ef']=data['wong1s']
data['ep']=data['ef']-data['wong2s']
data['np'] = data['Hamming']-data['ef']
data['nf'] = data['total'] - (data['ef']+data['ep']+data['np'])
data = data.drop_duplicates()

data['efXnp']= data['ef']*data['np']
data['epXnf']=data['nf']*data['ep']
data['col'] = np.where(data['ep'] != 0, (((data['num_vars']+data['num_args']+data['max_age']+data['churn']+data['ef'])*(data['ep']))), 9000)
data['varsXargs'] = data['num_vars']+data['num_args']
data['nf'].fillna(0, inplace=True)
X = data[['methodId','ef','ep','nf','np','col']]
data = data.sort_values('bid')
filtered_df = data[data['efXnp'] > data['epXnf']]
orders = {}
for key in arr:
    filtered_df = data.sort_values(by=[key], ascending=[False])
    filtered_df = filtered_df.reset_index()
    filtered_df[[key,'label']].to_csv('check'+key+'.csv', index=False)
    
    indices = filtered_df.index
    # print(indices)
    indices = indices[::-1]
    ans = 0
    for ind in indices:
        if filtered_df.loc[ind, 'label']:
            ans = ind
            break
    orders[key] = ans

print(orders)
order_df = pd.DataFrame([orders])
order_df.to_csv('order.csv')
filtered_df[[arr[1],'label']].to_csv('check.csv', index=False)
# y = data.iloc[:,41]
# color = ['red' if x else 'blue' for x in data['buggy']]
# X = [y if x else -1 for x, y in zip(data['buggy'], data['ef'])]
# Y = [y if x else -1 for x, y in zip(data['buggy'], data['ep'])]
# print(type(X),type(Y))
# plt.scatter(X,Y)

# # Adding labels and title
# plt.xlabel('ef')
# plt.ylabel('ep')
# plt.title('Points on a Cartesian Plane')

# Display the plot
# plt.show()

# # Initializing the K-fold cross-validator
# true_indexes = data.loc[data['buggy'] == True].index
# false_indexes = data.loc[data['buggy'] == False].index

# confusion_matrices = []

# label = [True, False]

# def bug_accuracy(Y_true,Y_pred,lenght):
#     correctly_predicted = 0

#     # iterating over every label and checking it with the true sample
#     for true_label, predicted in zip(Y_true, Y_pred):
#         # print(true_label, predicted)
#         if true_label==1 and  predicted== 1:
#             correctly_predicted += 1
#     # computing the accuracy score
#     accuracy_score = correctly_predicted / lenght
#     return accuracy_score


# for i in range(5):
#     true_index_list = true_indexes.tolist()
#     np.random.seed(100)
#     np.random.shuffle(true_index_list)
#     split_point = int(len(true_index_list) * 0.8)
#     true_data = data.iloc[true_index_list]
#     true_train = true_data[:split_point]
#     true_test = true_data[split_point:]

#     false_index_list = false_indexes.tolist()
#     np.random.shuffle(false_index_list)
#     split_point = int(len(false_index_list) * 0.8)
#     true_data = data.iloc[false_index_list]
#     false_train = true_data[:split_point]
#     false_test = true_data[split_point:]


#     X_train = pd.concat([true_train,false_train],ignore_index=True)[['ef','ep','nf','np']]
#     y_train = pd.concat([true_train,false_train],ignore_index=True).iloc[:, 41]

#     X_test = pd.concat([true_test,false_test],ignore_index=True)[['ef','ep','nf','np']]
#     y_test = pd.concat([true_test,false_test],ignore_index=True).iloc[:, 41]




#     model =  model = BalancedRandomForestClassifier(n_estimators=100)
#     model.fit(X_train, y_train)

#         # Predicting on the test set
#     y_pred = model.predict(X_test)
#     class2_mask = y_test == True

# # Calculate the accuracy for class 2
#     score = bug_accuracy(y_test,y_pred,len(y_test.loc[data['buggy'] == True]))


#     # print(score)

#         # Calculating the confusion matrix
#     cm = confusion_matrix(y_true=y_test, y_pred=y_pred, labels=label)
#     TN, FP, FN, TP = cm.ravel()

#         # Calculating accuracy, precision, and recall
#     accuracy = (TP + TN) / (TP + TN + FP + FN)
#     if TP == 0 and FP == 0:
#             precision = 0.0
#     else:
#             precision = TP / (TP + FP)

#     if TP == 0 and FN == 0:
#             recall = 0.0
#     else:
#             recall = TP / (TP + FN)
#     confusion_matrices.append(cm)


#     print(f"Confusion matrix for Fold {len(confusion_matrices)}:")
#     print(cm)
#     print(TN/(TN+FP))
#     print("accuracy : ",accuracy, "precision :", precision, 'recall :',recall, '\n\n')
#     print()
