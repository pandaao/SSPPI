import numpy as np
from paddle.io import Dataset
from generate_matrix import generatematrix
import math
def position_encoding(pos, dmoudle):
    position = []
    for i in range(dmoudle):
        d = 2*i
        position.append(math.sin(pos / math.pow(10000, d / (dmoudle*2))))
        position.append(math.cos(pos / math.pow(10000, (d+1) / (dmoudle*2))))
    return position


class dataset(Dataset):
    def __init__(self, pdb_id, length, dict_seq, dict_pdb):
        """
        数据读取器
        :param data_path: 数据集所在路径
        :param mode: train or eval
        """
        super().__init__()
        self.pdb_id = pdb_id
        self.length = length
        self.input_matrix_sum = []
        self.labels = []
        self.input_index_sum = []
        self.dict_seq = dict_seq
        self.dict_pdb = dict_pdb
        n = 2
        input_matrix, input_label= generatematrix(pdb_id, self.dict_pdb, self.dict_seq)
        np_expansion = np.zeros((self.length, 6, 20)).astype(float)
        input_matrix = np.concatenate((np.concatenate((np_expansion, input_matrix), axis=0), np_expansion),
                                      axis=0).astype(float)
        np_expansion = np.zeros((self.length, 256)).astype(float)
        input_index = []
        for i in range(len(input_label)):
            input_index.append(position_encoding(i+1,128))
        input_index = np.concatenate((np.concatenate((np_expansion, input_index), axis=0), np_expansion), axis=0).astype(float)
        for i in range(len(input_label)):
        # print(np.shape(input_matrix[0:0+30]), input_label[0], np.shape(input_index[0:0+30]))
            self.input_matrix_sum.append(input_matrix[i:i + 2 * self.length + 1])
            self.labels.append([input_label[i]])
            self.input_index_sum.append(input_index[i:i + 2 * self.length + 1])

    def __getitem__(self, index):
        """
        获取一组数据
        :param index: 索引号
        :return:
        """
        return self.input_matrix_sum[index], self.labels[index], self.input_index_sum[index]

    def __len__(self):
        return len(self.labels)
