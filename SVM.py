import os

import pandas as pd
import numpy as np
from prettytable import PrettyTable
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn import metrics
import numpy
from keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_auc_score


os.environ["CUDA_VISIBLE_DEVICES"]="0"
numberList = [10,20,30,40,50,60,70,80,90,100]
factor_to_use = pd.read_csv(r"C:\\Users\\office\\Desktop\\data\\factor_importance_new0201.csv", usecols=[0],
                                encoding="gbk")
df = pd.read_csv(r"C:\\Users\\office\\Desktop\\data\\factorData\\data1.csv")
for factorNumber in numberList:
    factorList = factor_to_use.values[0:factorNumber, :]
    data = pd.DataFrame(columns=['level'], data=df["level"].values.reshape(-1, 1))
    for factor in factorList:
        data[factor[0]] = df[factor[0]].values.reshape(-1, 1)
    data = data.fillna(method='ffill')  # ffill指用向前法填充缺失值
    data = np.array(data)

    data_x = data[:, 1:]
    data_y = data[:, 0]
# build and evaluate different algorithms
    model_lg = LogisticRegression(class_weight = 'balanced') # logistic Regression
    model_dt = DecisionTreeClassifier(random_state=0) # Decision Tree
    model_rf = RandomForestClassifier(n_estimators=factorNumber, random_state=0) # Random Forest: 50 trees
    model_ada = AdaBoostClassifier(n_estimators=factorNumber, random_state=0) # Adaboost: 50 trees
    model_svm = SVC(kernel='linear', class_weight = 'balanced')

    models = []
    # models.append(["lg", model_lg])
    # models.append(["dt", model_dt])
    # models.append(["rf", model_rf])
    # models.append(["ada", model_ada])
    models.append(["svm", model_svm])

    stock_lenth = data.shape[0]
    tr_val_slip = tr_val_slip = int(0.8 * stock_lenth)  # 测试集中数据组数
    train_x = data_x[0:tr_val_slip]
    train_y = data_y[0:tr_val_slip]
    valid_x = data_x[tr_val_slip:]
    valid_y = data_y[tr_val_slip:]

    print("====================factorNum=", factorNumber, "=========================")

    for name, model in models:
        model.fit(train_x, train_y)
        val_predict = numpy.asarray(model.predict(valid_x))
        val_targ = numpy.asarray(valid_y)
        val_auc = []
        val_f1s = []
        val_recalls = []
        val_precisions = []
        val_accuracy = []
        _val_auc = roc_auc_score(val_targ, val_predict)
        _val_f1 = f1_score(val_targ, val_predict)
        _val_recall = recall_score(val_targ, val_predict)
        _val_precision = precision_score(val_targ, val_predict)
        _val_accuracy = accuracy_score(val_targ, val_predict)

        table = PrettyTable(["epoch", "accuracy_score", "precision_score", "recall_score", "F1_score", "AUC_score"])
        table.add_row([1, round(_val_auc, 4), round(_val_f1, 4),
                       round(_val_recall, 4), round(_val_precision, 4),
                       round(_val_accuracy, 4)])
        print(table)

    print("====================factorNum=", factorNumber, "=========================")