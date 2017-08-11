'''

csv 파일을 신경망으로 학습할 수 있게 하는 클래스.

csv파일을 불러와서 넘파이 배열로 변환.(원핫인코딩 혹은 정규화)

'''


import csv,numpy as np

class loaddata:

    def __init__(self,dir):
        self.dir = dir
        self.filename = ['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5']

    def csvimport(self,filedir):
        file = open(filedir,'r')
        batch = csv.reader(file)
        csv_file = []
        for i in batch:
            csv_file.append(i)
        return csv_file

    def one_hot_encoding(self,data_batch,single_data=True):
        if single_data:
            a = np.zeros([50000, 10])
        else:
            a = np.zeros([10000, 10])
        for idx, value in enumerate(data_batch):
            a[int(idx)][int(value[1024])] = 1
        return a

    def normalization(self,data):
        return data / 255

    def _loaddata(self,*filename,one_hot_encoding=True,normalization=True,single_data=False):
        if single_data:
            for idx,value in enumerate(self.filename):
                dir = self.dir + value + '.csv'
                if idx<=0:
                    temp = self.csvimport(dir)
                    x_data = temp
                else:
                    temp = self.csvimport(dir)
                    x_data = np.vstack((x_data,temp))
            if one_hot_encoding:
                x_label = self.one_hot_encoding(x_data,single_data=True)
                if normalization:
                    return self.normalization(np.array(x_data)[:,0:1024].astype(np.float32)), np.array(x_label)
                else:
                    return np.array(x_data)[:, 0:1024].astype(np.float32), np.array(x_label)
            else:
                x_label = np.array(x_data)[:, 1024]
                if normalization:
                    return self.normalization(np.array(x_data)[:, 0:1024].astype(np.float32)), x_label
                else:
                    return np.array(x_data)[:, 0:1024].astype(np.float32), x_label

        else:
            x_data = self.csvimport(self.dir+filename[0]+'.csv')
            if one_hot_encoding:
                x_label = self.one_hot_encoding(x_data,single_data=False)
                if normalization:
                    return self.normalization(np.array(x_data)[:,0:1024].astype(np.float32)), np.array(x_label)
                else:
                    return np.array(x_data)[:, 0:1024].astype(np.float32), np.array(x_label)
            else:
                x_label = np.array(x_data)[:,1024]
                if normalization:
                    return self.normalization(np.array(x_data)[:,0:1024].astype(np.float32)), x_label
                else:
                    return np.array(x_data)[:, 0:1024].astype(np.float32), x_label



# load = loaddata('D:\\Python\\PF\\DATA\\rawdata\\') #저장경로 입력
# x_data, x_label = load._loaddata(one_hot_encoding=True,normalization=True,single_data=True) #one-hot 형태로. 모든 데이터 통합해서.




