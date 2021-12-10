import os
import numpy as np
from .ECGXMLReader import ECGXMLReader

# 일단은 하드 코딩 - 나중에 이 파일들 어떻게 찾았는지 추가//한번 다 돌려보고 에러나는 파일명 추출해서 수작업으로 삭제했어요...
# 주석도 나중에 정리하기!
starts_w6_t = ['1267', '1970', '3131', '3198', '3469', '3618', '4827', '4971', '4979', '5055']
starts_w8_t = ['1879', '2164', '5580']
starts_w8_v = ['7226', '7281', '8783']


class DataPreprocess:

    def __init__(self):
        self.name_list = []
        self.train_path_arr = './data/train/arrhythmia/'
        self.train_path_nor = './data/train/normal/'
        self.val_path_arr = './data/validation/arrhythmia/'
        self.val_path_nor = './data/validation/normal/'

        self.train_arr_list = os.listdir(self.train_path_arr)
        self.train_nor_list = os.listdir(self.train_path_nor)
        self.val_arr_list = os.listdir(self.val_path_arr)
        self.val_nor_list = os.listdir(self.val_path_nor)
        self.train_arr_list.sort()
        self.train_nor_list.sort()
        self.val_arr_list.sort()
        self.val_nor_list.sort()

        # 오류 나는 파일명 제거
        # 깃허브에 올릴 때 우린 데이터를 빼고 올리지만, 심사하는 분들은 전체 데이터가 있는 상태에서 구동시키실 테니까
        # 이 코드를 추가해봤는데 오류 나는 파일들을 어떤 방법으로 찾았는지는 데이터 처리 작업을 직접 한 채은이가 설명을 추가해줘!!
        self.remove_possible_errors(self.train_arr_list)
        self.remove_possible_errors(self.val_arr_list, False)

        self.decode_files()
        self.save_npy()

    @staticmethod
    def remove_possible_errors(files, is_train_set=True):
        if is_train_set:
            for index in starts_w6_t:
                file = f'6_2_00{index}_ecg.xml'
                files.remove(file)
            for index in starts_w8_t:
                file = f'8_2_00{index}_ecg.xml'
                files.remove(file)
        else:
            for index in starts_w8_v:
                file = f'8_2_00{index}_ecg.xml'
                files.remove(file)

    @staticmethod
    def decode_file(filepath, names):
        file = []
        for name in names:
            file.append(ECGXMLReader(filepath + name, name, augmentLeads=True))
        return file

    @staticmethod
    def find_weirdos(decoded):  # inputs: train_arr, train_nor, val_arr, val_nor/길이가 이상한 리스트 뽑기
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
    def making_arr(data_list, diff_list):  # train_arr, train_nor, val_arr, val_nor넣고 돌리기.4096길이로 자르기.
        train_array = []
        for i, file in enumerate(data_list):
            if i in diff_list:  # 앞에서 추출한 크기가 상이한 데이터들을 제외하고 array를 생성했습니다.
                continue
            l = []
            arr = sorted(file.getAllVoltages().items())
            for line in arr:
                l.append(line[1])  # 나중에 여기서 변수를 받아서 선택적으로 리드를 뽑게 수정가능
            l = np.asarray(l, dtype=object).astype('float32')
            l = l[:, :4096] 
            train_array.append(l)
        return train_array  # list로 return

    def decode_files(self):
        print("Decoding in process... ", end='')
        train_arr = self.decode_file(self.train_path_arr, self.train_arr_list)
        train_nor = self.decode_file(self.train_path_nor, self.train_nor_list)  # 나중에 train끼리 합체
        val_arr = self.decode_file(self.val_path_arr, self.val_arr_list)
        val_nor = self.decode_file(self.val_path_nor, self.val_nor_list)  # val 끼리 합체
        self.name_list = [train_arr, train_nor, val_arr, val_nor]
        print("Done!")

    # 향후 계획: list끼리 + --> 최종 trainset data 만들기. 향후 파이토치 모델에 넣을 시 shuffle의 작업이 꼭 필요함.

    def save_npy(self):
        print("Saving data as .npy... ", end='')
        diff_set = []
        for x in self.name_list:
            diff_set.append(self.find_weirdos(x))
        # diff_set 출력 결과로 미루어 보아, train dataset에서만 이상한 길이들이 존재함!

        ## making array, save as a file.npy
        train_arr_arraylist = self.making_arr(self.name_list[0], diff_set[0])
        train_nor_arraylist = self.making_arr(self.name_list[1], diff_set[1])
        val_arr_arraylist = self.making_arr(self.name_list[2], diff_set[2])
        val_nor_arraylist = self.making_arr(self.name_list[3], diff_set[3])

        train_array = np.asarray(train_arr_arraylist + train_nor_arraylist)
        val_array = np.asarray(val_arr_arraylist + val_nor_arraylist)

        if not os.path.isdir('./data_np/'):
            os.mkdir('./data_np/')

        np.save('./data_np/train', train_array, allow_pickle=True, fix_imports=True)
        np.save('./data_np/validation', val_array, allow_pickle=True, fix_imports=True)

        ## label의 경우, 주신 json파일에서 추출후 변형하는 방법도 있겠지만, 주신 데이터가 아예 정상, 비정상이 따로 왔기 때문에 그냥 0, 1로 데이터 개수에 맞춰 리스트를 만든 후, 합쳐서 배열을 만들었습니다.

        if not os.path.isdir('./label_np/'):
            os.mkdir('./label_np/')

        train_arr_label = [1] * (len(self.train_arr_list) - len(diff_set[0]))
        train_nor_label = [0] * (len(self.train_nor_list) - len(diff_set[1]))
        val_arr_label = [1] * (len(self.val_arr_list) - len(diff_set[2]))
        val_nor_label = [0] * (len(self.val_nor_list) - len(diff_set[3]))
        train_label = np.asarray(train_arr_label + train_nor_label).astype('float32').reshape(-1, 1)
        val_label = np.asarray(val_arr_label + val_nor_label).astype('float32').reshape(-1, 1)
        np.save('./label_np/train', train_label, allow_pickle=True, fix_imports=True)
        np.save('./label_np/validation', val_label, allow_pickle=True, fix_imports=True)
        print("Done!")
