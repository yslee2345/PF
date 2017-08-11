'''
cifar10의 최초 데이터를 csv 파일 형태로 변환 *그레이 스케일


'''

import pickle,csv,numpy as np

class preprocessing:

    R = 0.2126
    G = 0.7152
    B = 0.0722

    def __init__(self,path):
        self.datawithlabels = []
        self.data = None
        self.labels = None
        self.path = path

    def unpickle(self,file):
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def loaddata(self):
        batch = self.unpickle(str(self.path)+str(self.filename))
        self.data = batch[b'data']
        self.labels = batch[b'labels']

    def grayscale(self):
        data = self.data[:,0:1024]*preprocessing.R + self.data[:,1024:2048] *preprocessing.G + self.data[:,2048:] * preprocessing.B
        self.data = np.array(data).astype('int32')

    def writecsv(self,grayscale=True):
        with open(self.path+self.filename+'.csv','w',newline='') as csvfile:
            writer = csv.writer(csvfile,delimiter=',')
            if grayscale:
                self.grayscale()
            for i,j in zip(self.data,self.labels):
                writer.writerow(np.append(i,j))

    def preprocess(self,filename,grayscale=True):
        self.filename = filename
        self.loaddata()
        print(self.filename + '-----------------Data load Complete----------------------')
        self.writecsv(grayscale=grayscale)
        print(self.filename + '-----------------Preprocessing Complete------------------')

if __name__=='__main__':
    pre = preprocessing('D:\\Python\\PF\\DATA\\rawdata\\')
    pre.preprocess('data_batch_1',grayscale=True)
    pre.preprocess('data_batch_2',grayscale=True)
    pre.preprocess('data_batch_3',grayscale=True)
    pre.preprocess('data_batch_4',grayscale=True)
    pre.preprocess('data_batch_5',grayscale=True)
    pre.preprocess('test_batch',grayscale=True)