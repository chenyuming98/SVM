import cv2
import os
import numpy as np
import re

if __name__ == '__main__':
    print("begin")
    bowTrainer = cv2.BOWKMeansTrainer(3)
    dic = {}
    train_set = {}  #训练集
    train_path = os.getcwd() + "\\file\\train\\"
    match_path = os.getcwd() + "\\file\\match\\"
    lab_num = 0
    lab_mat =[]
    train_list = [] #svm训练向量集
    for filename in os.listdir(train_path):
        i = 0
        cFile = train_path + filename
        ls = []
        la = os.listdir(cFile)
        print( filename + "：类训练文件统计："+ str(len(la)) )
        print("标签为：" + str(lab_num))
        for imgName in os.listdir(cFile):
            imgPath = imgFile = cFile + "\\" + imgName
            img = cv2.imread(imgPath)
            if img is None:
                print("图像为空")
                print("图像为空的数据：" + cFile + "\\" +imgName)
                continue
            img = cv2.resize(img, (200, 200))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # cv2.imshow(imgPath, img)
            ls.insert(i,imgPath)
            train_list.append(img)
            lab_mat.append([lab_num])  # svm 样本标签向量
        dic[filename] = ls
        lab_num += 1
    #转成np.array
    train_list = np.array(train_list).astype(np.float32)
    lab_mat = np.array(lab_mat)

    train_list = train_list.reshape(-1,200*200)
    # k = cv2.waitKey(20000000)
    #建立svm及參數
    SVM = cv2.ml.SVM_create()
    SVM.setType(cv2.ml.SVM_C_SVC)
    SVM.setKernel(cv2.ml.SVM_LINEAR)


    # 开始训练
    print("开始训练SVM模型")
    SVM.train(train_list, cv2.ml.ROW_SAMPLE, lab_mat)
    print("***********************")
    print("训练结束开始预测")
    match_list = []
    mFile = match_path
    match_name = []
    for imgName in os.listdir(mFile):
        imgPath = imgFile = mFile + "\\" + imgName
        img = cv2.imread(imgPath)
        img = cv2.resize(img, (200, 200))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        match_list.append(img)
        match_name.append(imgName)
    match_list = np.array(match_list).astype(np.float32)
    match_list = match_list.reshape(-1,200*200)

    print("预测分类结果")
    result = SVM.predict(match_list)
    # print(result)
    countNum = 0
    matchResult = {}
    for score in result[1]:
        matchResult[match_name[countNum]] = int(score)
        countNum += 1
    print(matchResult)

    print("end")


    # print( correct * 100.0 / result[1].size)