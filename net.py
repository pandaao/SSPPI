import os
import random
import paddle
from dataset import dataset
from runNet import runNet
from generate_matrix import Legitimacy_check, getdict
import numpy as np
import math
from MyNet import MyNet
'''
参数配置
'''
train_parameters = {
    "target_path": "oridata/",  
    "train_list_path": "data/train.txt",  # train.txt路径
    "eval_list_path": "data/eval.txt",  # eval.txt路径
    "skip_steps": 20,
    "save_steps": 100,
    "learning_strategy": {  # 优化函数相关的配置
        "lr": 0.001  # 超参数学习率
    },
    "checkpoints": "work/checkpoints"  # 保存的路径
}

dict_seq, dict_pdb = getdict()
def get_data_list(target_path, train_list_path, eval_list_path, dict_seq, dict_pdb):
    """
        生成数据列表
    """
    Legitimacy_check(dict_pdb, dict_seq)
    
   
    targetdir = os.listdir(target_path)
    trainer_list = []
    pdb_num = 0
    for pdb in targetdir:
        if (pdb.split('.')[0] + '\n') not in defectivepdb:
                trainer_list.append(pdb)

    #打乱顺序

    random.shuffle(trainer_list)
    with open(train_list_path, 'a') as f:
        for pdb in trainer_list:
            f.write(pdb.split('.')[0]+'\n')
    print('数据列表生成完毕！')

target_path = train_parameters['target_path']
train_list_path = train_parameters['train_list_path']
eval_list_path = train_parameters['eval_list_path']

# 每次生成数据列表前，首先清空train.txt和eval.txt
with open(train_list_path, 'w') as f:
    f.seek(0)
    f.truncate()
with open(eval_list_path, 'w') as f:
    f.seek(0)
    f.truncate()
# 生成数据列表

get_data_list(target_path, train_list_path, eval_list_path, dict_seq, dict_pdb)
dict_eval = {}
L = 3
    # 数据加载
accs_sum = []
with open(os.path.join('data', "train.txt"), "r", encoding="utf-8") as f:
    info_train = f.readlines()
with open(os.path.join('data', "eval.txt"), "r", encoding="utf-8") as f:
    info_eval = f.readlines()

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
            if sit[i] == '0':
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

class runNet_eval():
    def __init__(self,train_parameters,loader,length, index=None):
        super(runNet_eval, self).__init__()
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




optimalauc = 0
dict_mcc = {}
for j in range(30):
    
    accs = []
    precisions = []
    recalls = []
    f1s = []
    mccs = []
    mcc = 0.3
    
    for i in range(len(info_train)):
        pdb_train_id = info_train[i].strip()
        if pdb_train_id in dict_mcc:
            mcc = dict_mcc.get(pdb_train_id)
            if mcc == 0.0:
                continue
        else:
            dict_mcc.update({pdb_train_id:mcc})
        try:
            train_dataset = dataset(pdb_train_id, L, dict_seq, dict_pdb)
            input_label = dict_pdb.get(pdb_train_id)
            lab = np.array(list(input_label))
            len0 = len(np.where(lab == '0')[0])
            len1 = len(np.where(lab == '1')[0])
            if len1 != 0:
                train_loader = paddle.io.DataLoader(train_dataset, batch_size=len0+len1, shuffle=True)
                paddle.device.set_device("gpu:0")
                myNet = runNet(train_parameters, train_loader, L, i)
                mcctmp = myNet.train(pdb_train_id, len0, len1, mcc)
                if mcctmp != mcc:
                    dict_mcc.update({pdb_train_id:mcctmp})
            else:
                continue
        except:
            continue
    eval_txt = ['Dset_72.txt', 'Dset_164.txt','Dset_186.txt','Dset_355.txt','Dset_448.txt']
    for l in range(5):
        with open(os.path.join('data', eval_txt[l]), "r", encoding="utf-8") as f:
            info_eval = f.readlines()
        label = ''
        site = ''
        y_true = []
        y_pred = []
        f1 = open('evaluating/'+eval_txt[l], "a+", encoding="utf-8")
        for i in range(len(info_eval)):
            try:
                pdb_eval_id = info_eval[i].strip()
                eval_dataset = dataset(pdb_eval_id, L, dict_seq, dict_pdb)
                input_label = dict_pdb.get(pdb_eval_id)
                lab = np.array(list(input_label))
                eval_loader = paddle.io.DataLoader(eval_dataset, batch_size=len(lab), shuffle=True)
                paddle.device.set_device("gpu:0")
                myNet = runNet_eval(train_parameters, eval_loader, L)
                label, site, y_true, y_pred = myNet.evalue_auc(pdb_eval_id, label, site, y_true, y_pred)

            except:
                f1.write(pdb_eval_id)
                continue
        from sklearn.metrics import roc_curve, auc, average_precision_score, confusion_matrix, precision_score, recall_score,  roc_auc_score, f1_score
        y_true_1 = np.array(y_true).argmax(axis=1)
        y_pred_1 = np.array(y_pred).argmax(axis=1)
        fpr, tpr,_ = roc_curve(y_true_1, y_pred_1)
        TP, TN, FP, FN = getevalute(label, site)
        accs, precisions, recalls, f1s, mccs =  compute_evalute(TP,TN,FP,FN)
        print(eval_txt[l])
        print("AUROC: {}".format(auc(fpr,tpr)))
        print("AUPRC: {}".format(average_precision_score(y_true_1, y_pred_1)))
        print("Accuracy:{}".format(accs))
        print("Precision:{}".format(precisions))
        print("Recall:{}".format(recalls))
        print("F1:{}".format(f1s))
        print("MCC:{}".format(mccs))
        strofans = str(auc(fpr,tpr))+"\t"+str(average_precision_score(y_true_1, y_pred_1))+"\t"+str(accs)+"\t"+str(precisions)+"\t"+str(recalls)+"\t"+str(f1s)+"\t"+str(mccs)+"\t"+str(TP)+"\t"+str(TN)+"\t"+str(FP)+"\t"+str(FN)
        f1.write(strofans+'\n')
        f1.close()
