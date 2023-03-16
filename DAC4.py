import os
import pandas as pd
import re
from sklearn import tree
from sklearn.metrics import accuracy_score
import argparse


def extract_feature(dataset):
    res_df = pd.DataFrame()
    label_folders = dataset
    labelfile = "labels_modified.txt"
    for resp in label_folders:
        label_data_path = os.path.join(resp[:-1], labelfile)
        if not os.path.exists(label_data_path):
            continue
        y_dataframe = pd.read_csv(label_data_path, sep=' ', header=None)
        y = y_dataframe[1]
        chipfaultinfo = resp.split("/")[1:4]
        # print(chipfaultinfo)
        assert len(chipfaultinfo)==3
        root = os.path.join("../", chipfaultinfo[0], chipfaultinfo[1], "tmax_fail", chipfaultinfo[2].split("_")[0]+"_fail")
        fail_data = "all.fail"
        fail_data_path = os.path.join(root, fail_data)
        x_dataframe = pd.read_csv(fail_data_path, sep=' ', header=None)
        
        if re.search(r'all.bmp', y_dataframe[0].iloc[-1]):
            # print("in ")
            y = y.iloc[:-1]

        x1 = x_dataframe[0]
        temp = []
        for num in x1:
            if len(temp)==0:
                temp.append(num)
            elif temp[-1]!=num:
                temp.append(num)
            else:
                continue
        x1 = pd.Series(temp)
        x2 = pd.Series(range(1, len(x1) + 1))
        x3 = [(x_dataframe.iloc[:, 0] <= x).values.sum() for x in x1]
        x3 = pd.Series(x3)
        x4 = [(x_dataframe.iloc[:, 0] == x).values.sum() for x in x1]
        x4 = pd.Series(x4)
        x5 = [len(set(x_dataframe.iloc[:, 1][x_dataframe.iloc[:, 0] <= x])) for x in x1]
        current = x5[0]
        x6 = [current]
        for i in range(1, len(x5)):
            x6.append(x5[i] - current)
            current = x5[i]
        x5 = pd.Series(x5)
        x6 = pd.Series(x6)
        x7 = [x1[0] for i in range(len(x1))]
        x7 = pd.Series(x7)
       
        data_with_label = pd.concat([x1, x2, x3, x4, x5, x6, x7, y], axis=1)
        # print(data_with_label)
        res_df = pd.concat([res_df, data_with_label], axis=0)
        
    return res_df


def calacc(model, testset):
    # print(testset)
    labelfile = "labels_modified.txt"
    guard = 3 # 保护带
    totalchip = 0
    correct = 0
    incorrect = 0
    id_resp = [-1]*len(testset)
    # print(testset)
    for p in range(len(testset)):
        if not os.path.exists(os.path.join(testset[p][:-1], labelfile)):
            # print(testset[p])
            continue
        totalchip += 1
        xy = extract_feature([testset[p]])
        xy.columns = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'y']

        x = xy.iloc[:, 0:-1]
        y = xy.iloc[:, -1]
        pred = model.predict(x)
        assert len(y)==len(pred)
        isStop = False
        # 预测标签全为0的情况，也算作correct
        if pred.sum()==0:
            correct+=1
            continue
        # for i in range(len(pred)):
        #     if pred[i]==0:
        #         continue
        #     elif pred[i]==1 :
        #         id_resp[p] = i
        #         if y[i]==1:
        #             correct += 1
        #             break
        #         else:
        #             incorrect += 1
        #             break
        for i in range(guard-1, len(pred)):
            if pred[i]==0:
                continue
            elif pred[i]==1:
                if sum(pred[i-guard:i+1])==1:
                    # 满足保护带机制
                    id_resp[p] = i
                    isStop = True
                    if y[i]==1:
                        correct+=1
                        break
                    else:
                        incorrect+=1
                        break
                else:
                    # 不满足保护带机制
                    continue
            else:
                raise Exception
        if not isStop:
            correct+=1
            continue

        

    if correct+incorrect==totalchip:
        return float(correct/totalchip), id_resp
    else:
        raise Exception
        return None


def calDVR(testset, id_resp):
    assert len(testset)==len(id_resp)
    dvr = 0.0
    for i in range(len(testset)):
        chipfaultinfo = testset[i].split("/")[1:4]
        if id_resp[i]==-1:
            continue
        failpath = os.path.join("../", chipfaultinfo[0], chipfaultinfo[1], "tmax_fail", chipfaultinfo[2].split("_")[0]+"_fail", f"{id_resp[i]}.fail")
        allfailpath = os.path.join("../", chipfaultinfo[0], chipfaultinfo[1], "tmax_fail", chipfaultinfo[2].split("_")[0]+"_fail", "all.fail")
        fenzi = len(open(failpath, "r").readlines())
        fenmu = len(open(allfailpath, "r").readlines())
        assert fenzi<=fenmu
        dvr = dvr + (1-(fenzi/fenmu))

    return dvr/len(testset)


def run(source, target):
    source_circuit = source
    target_circuit = target
    fault_types = ["ssl", "msl", "dom", "fe", "and", "or"]
    print("************************************")
    print(source_circuit, "-->", target_circuit)
    col_names = ['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'y']
    # source
    # trainset_s, testset_s = splitDataset(source_circuit, fault_types)
    with open(f"./dataset/{source_circuit}_train_list.txt", "r") as f:
        trainset_s = f.readlines()
    f.close()
    with open(f"./dataset/{source_circuit}_test_list.txt", "r") as f:
        testset_s = f.readlines()
    f.close()
    # trainset_s = ["pic/ctrl/fe/2_resp\n"]
    x_y_train = extract_feature(trainset_s)
    # print(x_y_train)
    x_y_train.columns = col_names

    x_train_s = x_y_train.iloc[:, 0:-1]
    y_train_s = x_y_train.iloc[:, -1]

    data_path = os.path.join("data/", target_circuit)
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    # x_y_train.to_csv(f'{data_path}/{target_circuit}_train.csv', index=False)

    # target
    # _, testset_t = splitDataset(target_circuit, fault_types)
    with open(f"./dataset/{target_circuit}_test_list.txt", "r") as f:
        testset_t = f.readlines()
    f.close()
    
    # Train
    dt_clf = tree.DecisionTreeClassifier(min_samples_leaf=3, random_state=10)
    dt_clf.fit(x_train_s, y_train_s)

    # debug
    # x_y1 = extract_feature(trainset_s)
    # x_y1.columns = col_names

    # x_test_s = x_y1.iloc[:, 0:-1]
    # y_test_s = x_y1.iloc[:, -1]
    # dt_predicted_s= dt_clf.predict(x_test_s)
    # print("DT tree acc : ", accuracy_score(y_test_s, dt_predicted_s))
    
    # Test
    # r = calacc(model=dt_clf, testset=testset_s)
    # print("---------------source---------------")
    # acc_s, id_resp_s = calacc(model=dt_clf, testset=testset_s)
    # dvr_s = calDVR(testset_s, id_resp_s)
    # print(f"DT tree acc: {acc_s}".ljust(50), f"DVR : {dvr_s*100}%".ljust(30), f"len of {source_circuit} : {len(testset_s)}")
    # met_dict = cal_metrics(y_test_s, dt_predicted_s, labels=[0, 1])
    # print(met_dict)
    # print("---------------target---------------")
    acc_t, id_resp_t = calacc(model=dt_clf, testset=testset_t)
    dvr_t = calDVR(testset_t, id_resp_t)
    print(f"DT tree acc: {acc_t}".ljust(50), f"DVR : {dvr_t*100}%".ljust(30), f"len of {target_circuit} : {len(testset_t)}")
    # met_dict_t = cal_metrics(y_test_t, dt_predicted_t, labels=[0, 1])
    # print(met_dict_t)
    # print("************************************")
    print("\n")


if __name__ == '__main__':
    # chiplist = ["x1", "pair", "frg2", "i10", "des"]
    # chiplist = ["int2float", "dec", "priority", "sin", "ctrl", "cavlc", "i2c", "adder", "bar"]
    # chiplist = ["pair", "x1", "i10"]
    # for i in chiplist:
    #     for j in chiplist:
    #         if i==j:
    #             continue
    #         run(source=i, target=j)
    parser = argparse.ArgumentParser(description="decision tree")
    parser.add_argument("-s", type=str, default="ctrl", help="source domain")
    parser.add_argument("-t", type=str, default="ctrl", help="target domain")
    args = parser.parse_args()
    print(args)
    run(source=args.s, target=args.t)
    


