# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 02:08:23 2021

@author: User
"""

#參考:https://blog.techbridge.cc/2019/09/21/how-to-use-python-tkinter-to-make-gui-app-tutorial/;https://www.cnblogs.com/shwee/p/9427975.html
import tkinter as tk
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import sys
sys.path.append(r'..\library')
from local_utils import detect_lp,detect_lp2
from os.path import splitext,basename
from keras.models import model_from_json
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from plot import predict_probability
import glob
import argparse
#%%
def load_model(path): #載入模型
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path) #加載權重
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)
		
#處理圖片
# %%
def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path) #讀取圖片
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) #img圖片BGR格式轉換成RGB格式
    img = img / 255 #大於1會變白
    if resize:
        img = cv2.resize(img, (224,224)) #圖片縮放
        #cv2.imshow("resize",img)
    return img

def get_plate(image_path, Dmax=608, Dmin=508): 
    vehicle = preprocess_image(image_path) #縮放過的照片
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2]) #numpy 獲取比率 大的除小 提取前面兩個數
    side = int(ratio * Dmin) #288是最佳值 
    bound_dim = min(side, Dmax) #608是最佳值
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5) # CNN探測車牌
    return vehicle, LpImg, cor
 
def get_plate2(image_path, Dmax=608, Dmin=288):  #取其他方式獲取車牌
    #vehicle = preprocess_image(image_path) 
    vehicle = cv2.imread(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2]) #numpy 獲取比率 大的除小 提取前面兩個數
    side = int(ratio * Dmin) #288是最佳值 
    bound_dim = min(side + (side%(2**4)),608) #608是最佳值
    #print(max(vehicle.shape[:2]))
    #print(min(vehicle.shape[:2]))
    #print(im2single(vehicle))
    Llp , LpImg, cor = detect_lp2(wpod_net, im2single(vehicle), bound_dim, lp_threshold=0.5) # CNN探測車牌
    #cv2.imshow("resize",Llp)
    return vehicle, LpImg, cor

def im2single(I):
	assert(I.dtype == 'uint8')
	return I.astype('float32')/255.
	
# %%
#離群值
def detect_outliers(data,threshold=1.5): #需修改偏差大於多少 1.5是目前能測到最低的非數字
    mean_d = np.mean(data) #平均
    std_d = np.std(data) #標準差
    outliers = []
    z_k = 99
    for y in data:
        z_score= (y - mean_d)/std_d  #Z分數
        #print(z_score)
        #print(y)
        if z_k > np.abs(z_score):
            k= y
            z_k = np.abs(z_score)
    for y in data:
        z_score= (y - mean_d)/std_d  #Z分數
        if np.abs(z_score) > threshold:
            if abs(k - y) > 10:
                #print(k)
                #print(y)
                outliers.append(y) 
    return outliers
#  預處理輸入圖像和帶模型的Pedict
def predict_from_model(image,model,labels, i):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    n = i
    return prediction
def predict_from_model2(image,model,labels, i):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    n = i
    predict_probability(image,model.predict(image[np.newaxis,:]),labels, n)
    print("-----------------------------------")
    return prediction
# 創建sort_contours（）函數以從左到右抓取每個數字的輪廓
def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts
# %%
"""
### Part 1:  從樣本圖像中提取車牌
"""
# %%
wpod_net_path = "../library/wpod-net.json" 
wpod_net = load_model(wpod_net_path)




def image_path(test_image_path):
    vehicle, LpImg,cor = get_plate(test_image_path)
    fig = plt.figure(figsize=(12,6)) #畫板
    grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig) #畫布
    fig.add_subplot(grid[0])
    plt.axis(False) #不畫XY線條
    plt.imshow(vehicle)
    grid = gridspec.GridSpec(ncols=2,nrows=1,figure=fig)
    fig.add_subplot(grid[1])
    plt.axis(False)
    #plt.imshow(LpImg[0])
    
    # %%
    """
    ## Part 2: Segementing license characters
    #分割字元
    """
    
    # %%
    if True: # 是否有一個可判別的字元
        #  縮放，計算絕對值並將結果轉換為8位(256種顏色)
        plate_image = cv2.convertScaleAbs(np.float32(LpImg[0]), alpha=(255.0))
        
        # 轉換為灰度並模糊圖像
        gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
        for i in range(gray.shape[0]):
            for j in range(gray.shape[1]):
                # 3：对不同的灰度值选择不同的颜色
                if gray[i, j] < 155:
                    plate_image[i, j, :] = [255, 255, 255]
                elif 140 >= gray[i, j] >= 15:
                    plate_image[i, j, :] = [0, 0, 0]
                else:
                    plate_image[i, j, :] = [0, 0, 0]
        #gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)                                     
        blur = cv2.GaussianBlur(gray,(7,7),0)
        
        #   門檻值 180最小門檻值 255最大門檻值 cv2.THRESH_BINARY_INV 演算法類型
        binary = cv2.threshold(blur, 180, 255,
                             cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        #返回指定形狀和尺寸(矩形) (矩陣) 定義一個3*3的十字結構元素
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    	#膨脹
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        
    #可視化結果   
    fig = plt.figure(figsize=(12,7))
    plt.rcParams.update({"font.size":18})
    grid = gridspec.GridSpec(ncols=2,nrows=3,figure = fig)
    plot_image = [plate_image, gray, blur, binary,thre_mor]
    plot_name = ["plate_image","gray","blur","binary","dilation"]
    
    for i in range(len(plot_image)):
        fig.add_subplot(grid[i])
        plt.axis(False)
        plt.title(plot_name[i])
        #if i == 0:
        #    plt.imshow(plot_image[i])
        #else:
        #    plt.imshow(plot_image[i],cmap="gray")
    
    #plt.savefig("threshding.png", dpi=300)
    
    # %%
    
    
    cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # 創建plat_image的副本版本“ test_roi”以繪製邊界框
    test_roi = plate_image.copy()
    
    # 初始化一個列表，用於添加字符圖像
    crop_characters = []
    test=[]
    # 定義字符的標準寬度和高度
    digit_w, digit_h = 150, 200
    
    ratioMin = 1.0
    ratioMax = 10.5 #數字1長寬比過高 所以ratiomax要調高
    
    
    #尋找Y過於偏離的值
    data = []
    for c in sort_contours(cont): #數字與字母
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        print(ratio)
        if ratioMin<=ratio<=ratioMax: #長寬比低於7.5且高於1.5
            if h/plate_image.shape[0]>0.5: #長大於車牌的一半
                data.append(y)
    cont_fail = detect_outliers(data, threshold=1.5)  
    print(cont_fail)
       
    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if ratioMin<=ratio<=ratioMax: #僅選擇定義比例的輪廓
            if h/plate_image.shape[0]>0.5: #選擇大於50%的輪廓  
                if y not in cont_fail:
                   
                    # Draw bounding box arroung digit number 繪製邊界框箭頭數字
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)
            
                    # Sperate number and gibe prediction  分離數字和預測
                    curr_num = thre_mor[y:y+h,x:x+w]
                    #test.append(curr_num)
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    test.append(curr_num)            
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #返回指定形狀和尺寸(矩形) (矩陣)
                    crop_characters.append(curr_num)
        
    
    print("Detect {} letters...".format(len(crop_characters)))
    
    fig = plt.figure(figsize=(10,6))
    plt.axis(False)
    #plt.imshow(test_roi)
    
    #plt.show()
    #plt.savefig('grab_digit_contour.png',dpi=300)
    
    # %%
    fig = plt.figure(figsize=(14,4))
    grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)
    
    for i in range(len(crop_characters)):
        fig.add_subplot(grid[i])
        plt.axis(False)
        plt.imshow(crop_characters[i],cmap="gray")
    plt.savefig("output/segmented_leter.png",dpi=300)    
    
    # %%
    """
    ##  加載預訓練的MobileNets模型並進行預測
    """
    
    # %%
    # Load model architecture, weight and labels 加載模型全重與標籤
    json_file = open('output/MobileNets_character_recognition.json', 'r') #MobileNets 字符識別
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("output/License_character_recognition.h5")
    print("[INFO] Model loaded successfully...")
    
    labels = LabelEncoder()
    labels.classes_ = np.load('output/license_character_classes.npy')
    print("[INFO] Labels loaded successfully...")
    
    # %%
    
    # %%
    fig = plt.figure(figsize=(15,3))
    cols = len(crop_characters)
    grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)
    
    final_string = ''
    for i,character in enumerate(crop_characters): #遍立crop_characters圖片
        fig.add_subplot(grid[i])
        title = np.array2string(predict_from_model(character,model,labels, i))
        plt.title('{}'.format(title.strip("'[]"),fontsize=20))
        final_string+=title.strip("'[]")
        plt.axis(False)
        plt.imshow(character,cmap='gray')
    
    print(final_string)
    plt.savefig('output/final_result.png', dpi=300)
    #plt.show()
    # %%
    for i,character in enumerate(crop_characters):
        title = np.array2string(predict_from_model2(character,model,labels,i))
    return final_string  