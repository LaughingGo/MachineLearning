# coding=gbk
# ��������Ҫ�Ŀ�
import pandas as pd
import numpy as np
import math
import random
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
import time
import matplotlib.pyplot as plt
plt.style.use('ggplot')

if __name__ == '__main__':
    # ��������
    dataSource = pd.read_csv('./Dataset/krkopt.data', sep=',', header=None)
    # ���ַ�������תΪ��ֵ�ͱ���������ԭʼ���ݵĵ�1,3,5���Ѿ�����ֵ�ͣ�����ֻ��ת����0,2,4��
    le = LabelEncoder()
    le.fit(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'])
    dataNumerical = dataSource
    dataNumerical.iloc[:][0] = le.transform(dataSource.iloc[:][0])
    dataNumerical.iloc[:][2] = le.transform(dataSource.iloc[:][2])
    dataNumerical.iloc[:][4] = le.transform(dataSource.iloc[:][4])

    # ����ѵ������
    train_data = dataNumerical.iloc[:, 0:6]
    train_target = dataNumerical.iloc[:, 6]
    # ��ԭʼ���ݴ��ң���ֹ����ϻ��߲���������������һ����������ѵ��������Լ���
    X_train, X_test, y_train, y_test = train_test_split(train_data, train_target, test_size=0.2, random_state=0)

    #################### ��ʼѵ�� ###################
    # ��¼ʱ��
    startTime = time.time()
    # ���ò�ͬ�����Ļ�ѧϰ�����е���ѵ����ͳ�ƶ�Ӧ����ȷ����ѵ��ʱ��
    estimatorNumbers = np.linspace(2, 40, 20,dtype=int)
    Accuracys = []
    CostTimes = []
    # estimatorNumber = 32
    for estimatorNumber in estimatorNumbers:
        bg_params = {'max_features':6,'oob_score':True,'bootstrap_features': False, 'n_jobs': -1,'n_estimators':estimatorNumber}
        outputs = ['draw', 'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
                   'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen']

        # Bag1ToNΪ�洢����ѧϰ�������飨��ά��������һ�Զ���ԣ���18��bangging������ɣ�ÿ��bagging���������ĳһ�ض����ѵ��������
        Bag1ToN=[]
        # �ֱ����ÿһ�����ݵ�ʵ��ѵ���ִ�
        realStageNumber=np.zeros(18)
        # ѵ����������
        trainNum = len(X_train)
        # IterativeBaggingֻ�����ڻع����������⣻
        # ���ｫ������������չ�ɶ���࣬����һ�Զ�ԭ�����������»���
        X_train_TwoClass = []
        for tt in range(trainNum):
            X_train_TwoClass.append(X_train.iloc[tt])
        # ���ÿһ���ѵ��һ��������
        for k in range(18):
            # ��ǰ�������
            classK= outputs[k]
            # ��ǰ����Ϊ1������Ϊ0
            y_train_TwoClasses = [1 if (y_train.iloc[tt]==classK) else 0 for tt in range(len(y_train)) ]
            # �������ѵ���ִ�
            stageNumber = 15
            # �в��ʼ��
            next_Y = y_train_TwoClasses
            # ��¼��ǰ���ķ��������ϣ���ÿһ�ֵķ��������
            Bag = []
            # ��ǰ��С�в�ƽ����
            minSumofSqu = sum([next_Y[x]*next_Y[x] for x in range(trainNum)])
            realStageNumber[k] = 0
            # �ִε���
            for n in range(stageNumber):
                # ��ǰѧϰ��
                BagTemp = BaggingRegressor(**bg_params)
                BagTemp.fit(X_train_TwoClass,next_Y)

                # �������(��֤��ƫ����)
                predict_Y = BagTemp.oob_prediction_
                next_Y = [next_Y[m]-predict_Y[m] for m in range(trainNum)]
                Bag.append(BagTemp)
                realStageNumber[k] += 1
                #��ǰ�вnew_Y��ƽ���ʹ�����ʷ��Сֵ��1.1��ʱ��ֹѭ��
                minSumofSquTemp = sum([next_Y[x] * next_Y[x] for x in range(trainNum)])
                if(minSumofSquTemp>1.1*minSumofSqu):
                    break
                else:
                    if (minSumofSquTemp <minSumofSqu):
                        minSumofSqu = minSumofSquTemp
                print("Classifier#", k+1,"---stage��",n+1)
            Bag1ToN.append(Bag)
            print("Classifier#",k+1,"done!")
        print("Training time:", time.time() - startTime,"(s)")
        # ѵ����ɣ���¼ʱ��
        endTime = time.time()
        CostTimes.append(endTime-startTime)
        #################### Ԥ��������������������㾫��###################

        # ��ʼ������
        predict_Y_test = np.zeros(len(X_test))
        predict_Y_test_sum = np.zeros([len(X_test), 18])
        max_Index = np.zeros(len(X_test))
        max_sumY = np.zeros(len(X_test))
        # ÿ������Ӧ�ķ����������ڲ���������Ԥ��
        for j in range(18):
            Bagj = Bag1ToN[j]
            # ÿ���в��ִε�Ԥ��ֵ�ۼ���Ϊ���յ�Ԥ��ֵ
            for i in range(int(realStageNumber[j])):
                predict_Y_testTemp = Bagj[i].predict(X_test)
                predict_Y_test = [predict_Y_test[j] + predict_Y_testTemp[j] for j in range(len(X_test))]
            for s in range(len(X_test)):
                # ÿ�������Ļع�Ԥ��ֵ����0.5���Ϊ1�࣬�����Ϊ0��
                if predict_Y_test[s] > 0.5:
                    predict_Y_test_sum[s][j] += 1
                    if (predict_Y_test_sum[s][j] > max_sumY[s]):
                        max_Index[s] = j
                        max_sumY[s] = predict_Y_test_sum[s][j]
                else:
                    for t in range(18):
                        if t != j:
                            predict_Y_test_sum[s][t] += 1
                            if (predict_Y_test_sum[s][t] > max_sumY[s]):
                                max_Index[s] = t
                                max_sumY[s] = predict_Y_test_sum[s][t]
        # ����18����������Ԥ�������������ͶƱ������ȷ������������Ϊ�����
        predict_Y_test_Final = [outputs[int(max_Index[u])] for u in range(len(y_test))]
        accuracy_test = metrics.accuracy_score(y_test, predict_Y_test_Final)
        Accuracys.append(accuracy_test)

        #################### Ԥ��ѵ�����������������㾫��###################
        predict_Y_train = np.zeros(len(X_train))
        predict_Y_train_sum = np.zeros([len(X_train), 18])
        max_Index = np.zeros(len(X_train))
        max_sumY = np.zeros(len(X_train))
        for j in range(18):
            Bagj = Bag1ToN[j]
            for i in range(int(realStageNumber[j])):
                predict_Y_trainTemp = Bagj[i].predict(X_train)
                predict_Y_train = [predict_Y_train[j] + predict_Y_trainTemp[j] for j in range(len(X_train))]
            for s in range(len(X_train)):
                if predict_Y_train[s] > 0.5:
                    predict_Y_train_sum[s][j] += 1
                    if (predict_Y_train_sum[s][j] > max_sumY[s]):
                        max_Index[s] = j
                        max_sumY[s] = predict_Y_train_sum[s][j]
                else:
                    for t in range(18):
                        if t != j:
                            predict_Y_train_sum[s][t] += 1
                            if (predict_Y_train_sum[s][t] > max_sumY[s]):
                                max_Index[s] = t
                                max_sumY[s] = predict_Y_train_sum[s][t]

        predict_Y_train_Final = [outputs[int(max_Index[u])] for u in range(len(y_train))]
        accuracy_train = metrics.accuracy_score(y_train, predict_Y_train_Final)

        print("accuracy_test:", accuracy_test)
        print("accuracy2:-train", accuracy_train)
        print("Estimator",estimatorNumber,"done!")

    # ���ƻ������������������ྫ�ȹ�ϵͼ
    plt.figure('���ྫ��')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ����������ʾ���ı�ǩ
    plt.title("IterativeBagging�������������������ྫ�ȹ�ϵͼ")
    plt.xlabel("������������")
    plt.ylabel("���ྫ��")
    plt.plot(estimatorNumbers, Accuracys, label='IterativeBagging')
    plt.legend(loc='lower right')

    # ���ƻ�������������������ʱ���ϵͼ
    plt.figure('ѵ��ʱ��')
    plt.rcParams['font.sans-serif'] = ['SimHei']  # ����������ʾ���ı�ǩ
    plt.title("IterativeBagging��������������������ʱ���ϵͼ")
    plt.plot(estimatorNumbers, CostTimes, label='IterativeBagging')
    plt.legend(loc='lower right')
    plt.xlabel("������������")
    plt.ylabel("����ʱ��")
    plt.show()

    # ��ӡ���
    print("Accuracys:", Accuracys)
    print("Costtime:", CostTimes)
