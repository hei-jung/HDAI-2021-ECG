import os
import numpy as np
from .ECGXMLReader import ECGXMLReader


class DataPreprocess:

    def __init__(self, path_arr, path_nor, data_filename='data', label_filename='label'):
        self.name_list = []
        self.path_arr = path_arr
        self.path_nor = path_nor
        self.data_filename = data_filename
        self.label_filename = label_filename

        self.arr_list = os.listdir(self.path_arr)
        self.nor_list = os.listdir(self.path_nor)
        self.arr_list.sort()
        self.nor_list.sort()

        self.decode_files()
        self.save_npy()

    @staticmethod
    def decode_file(filepath, names):
        file = []
        for name in names:
            file.append(ECGXMLReader(filepath + name, name, augmentLeads=True))
        return file

    @staticmethod
    def find_weirdos(decoded):  # 길이가 다른 리스트 뽑기
        diff_list = []
        for i, file in enumerate(decoded):
            tmp = []
            arr = sorted(file.getAllVoltages().items())
            for line in arr:
                tmp.append(line[1])
            tmp = np.asarray(tmp)
            if tmp.shape != (12, 5000) and tmp.shape != (12, 4999):
                # 크기가 (12, 5000), (12, 4999)가 아닌 데이터들의 인덱스 리스트를 추출
                diff_list.append(i)
        return diff_list

    @staticmethod
    def making_arr(data_list, diff_list):  # 데이터 길이를 4096이 되도록 자르기
        arr = []
        for i, file in enumerate(data_list):
            if i in diff_list:  # 앞에서 추출한 크기가 상이한 데이터들을 제외하고 array 생성
                continue
            l = []
            arr = sorted(file.getAllVoltages().items())
            for line in arr:
                l.append(line[1])
            l = np.asarray(l, dtype=object).astype('float32')
            l = l[:, :4096]
            arr.append(l)
        return arr

    def decode_files(self):
        print("Decoding in process... ", end='')
        data_arr = self.decode_file(self.path_arr, self.arr_list)
        data_nor = self.decode_file(self.path_nor, self.nor_list)

        self.name_list = [data_arr, data_nor]
        print("Done!")

    def save_npy(self):
        print("Saving data as .npy... ", end='')
        diff_set = []
        for x in self.name_list:
            diff_set.append(self.find_weirdos(x))

        data_arr_arraylist = self.making_arr(self.name_list[0], diff_set[0])
        data_nor_arraylist = self.making_arr(self.name_list[1], diff_set[1])

        data = np.asarray(data_arr_arraylist + data_nor_arraylist)

        np.save(self.data_filename, data, allow_pickle=True, fix_imports=True)

        # 정상, 비정상으로 분류되어 제공된 데이터 개수에 맞춰 정상=0, 비정상=1로 인코딩한 리스트를 만들어 배열 생성

        arr_label = [1] * (len(self.arr_list) - len(diff_set[0]))
        nor_label = [0] * (len(self.nor_list) - len(diff_set[1]))
        label = np.asarray(arr_label + nor_label).astype('float32').reshape(-1, 1)
        np.save(self.label_filename, label, allow_pickle=True, fix_imports=True)

        print("Done!")
