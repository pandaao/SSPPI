import os
import random
import paddle
from dataset import dataset
from generate_matrix import Legitimacy_check, getdict
import numpy as np
from MyNet import MyNet
import sys
import math

'''
参数配置
'''
train_parameters = {
    "target_path": "oridata/",  # 要解压的路径
    "train_list_path": "data/train.txt",  # train.txt路径
    "eval_list_path": "data/eval.txt",  # eval.txt路径
    "num_epochs": 10,  # 训练轮数
    "skip_steps": 10,
    "save_steps": 100,
    "learning_strategy": {  # 优化函数相关的配置
        "lr": 0.001  # 超参数学习率
    },
    "checkpoints": "work/checkpoints"  # 保存的路径
}

dict_seq, dict_pdb = getdict()

target_path = train_parameters['target_path']
train_list_path = train_parameters['train_list_path']
eval_list_path = train_parameters['eval_list_path']
dict_eval = {}

L = 5
# 数据加载
accs = []
precisions = []
recalls = []
f1s = []
mccs = []

pdb_eval_id = sys.argv[1]

def getevalute(label, site):
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    lab = np.array(list(label))
    sit = np.array(list(site))
    for i in range(len(lab)):
        if lab[i] == sit[i]:
            if lab[i] == '0':
                TN += 1
            else:
                TP += 1
        else:
            if lab[i] == '0':
                FN += 1
            else:
                FP += 1
    return TP, TN, FP, FN

def compute_evalute(TP, TN, FP, FN):
    accu = float(TP+TN)/float(TP+TN+FP+FN)
    try:
        prec = float(TP)/float(TP+FP)
    except:
        prec = 0
    try:
        recall = float(TP)/float(TP+FN)
    except:
        recall = 0
    try:
        F1 = (2*prec*recall)/(prec+recall)
    except:
        F1 = 0
    try:
        MCC = float(TP*TN - FP*FN) / math.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    except:
        if FP ==0 and FN == 0:
            MCC = 1
        else:
            MCC = 0

    return round(accu,3),round(prec,3),round(recall,3),round(F1,3),round(MCC,3)

TP = 0
TN = 0
FP = 0
FN = 0
label = ''
site = ''
y_true = []
y_pred = []
class runNet():
    def __init__(self,train_parameters,loader,length, index=None):
        super(runNet, self).__init__()
        self.train_parameters = train_parameters
        self.loader = loader
        self.length = length
        self.index = index

    def evalue_auc(self,pdb_eval_id, label, site, y_true, y_pred):
        '''
        模型评估
        '''
        model__state_dict = paddle.load('work/checkpoints/bssppi_save'+str(1+2*self.length)+'.pdparams')
        model_eval = MyNet(self.length)
        model_eval.set_state_dict(model__state_dict)
        model_eval.eval()
        for _, data in enumerate(self.loader()):
            x_data = data[0]
            tmplabel = data[1]
            y_data = paddle.to_tensor(np.array(data[1]).T,dtype='int64')
            index_data = data[2]
            predicts = model_eval([x_data, index_data])
            for i in range(len(predicts)):
                y_pred.append(np.array(predicts[i]))
                if tmplabel[0][i]=='1':
                    y_true.append([0,1])
                    label = label+'1'
                else:
                    label = label+'0'
                    y_true.append([1,0])
                if predicts[i][0] > predicts[i][1]:
                    site = site+'0'
                else:
                    site = site+'1'

        return label, site, y_true, y_pred


eval_dataset = dataset(pdb_eval_id, L, dict_seq, dict_pdb)
input_label = dict_pdb.get(pdb_eval_id)
lab = np.array(list(input_label))    
eval_loader = paddle.io.DataLoader(eval_dataset, batch_size=len(lab), shuffle=True)
paddle.device.set_device("gpu:0")
myNet = runNet(train_parameters, eval_loader, L)
label, site, y_true, y_pred = myNet.evalue_auc(pdb_eval_id, label, site, y_true, y_pred)
print(label)
print(site)
from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix, precision_score, recall_score,  roc_auc_score, f1_score

y_true = np.array(y_true).argmax(axis=1)
y_pred = np.array(y_pred).argmax(axis=1)
fpr, tpr,_ = roc_curve(y_true, y_pred)
TP, TN, FP, FN = getevalute(label, site)
accs, precisions, recalls, f1s, mccs =  compute_evalute(TP,TN,FP,FN)
print("AUROC: {}".format(auc(fpr,tpr)))
print("AUPRC: {}".format(average_precision_score(y_true, y_pred)))
print("Accuracy:{}".format(accs))
print("Precision:{}".format(precisions))
print("Recall:{}".format(recalls))
print("F1:{}".format(f1s))
print("MCC:{}".format(mccs))


