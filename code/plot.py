# -*- coding: utf-8 -*-
"""
Created on Mon May 31 13:12:46 2021

@author: user
"""
import matplotlib.pyplot as plt
import numpy as np

#字元預測機率
def predict_probability(image, predict, labels, n):
    i = 0
    character = []
    chance = []
    for j in range(34):
        if predict[0][j] > 0.0000001:
            print(labels.inverse_transform([i]))
            print(predict[0][j])
            x = np.array2string(labels.inverse_transform([i]))
            character.append(x.strip("'[]"))
            chance.append(float(predict[0][j]))
        i = i+1    
    #print(predict)
    predict_plt(image, character, chance, n)   

#畫圖
def predict_plt(image, character, chance, n):
    plt.cla()
    plt.clf()
    plt.close()
    x = np.arange(len(character))
    plt.subplot(121)  
    plt.bar(x, chance, color='blue')
    plt.xticks(x, character)
    plt.xlabel('character')
    plt.ylabel('chance')
    plt.title('predict')
    plt.subplot(122)
    plt.imshow(image)
    plt.axis(False)
    plt.savefig("output/test {} .png".format(n), dpi=300)       


def draw_box(image_path, cor, thickness=3): 
    pts=[]  
    x_coordinates=cor[0][0]
    y_coordinates=cor[0][1]
    # store the top-left, top-right, bottom-left, bottom-right 
    # of the plate license respectively
    for i in range(4):
        pts.append([int(x_coordinates[i]),int(y_coordinates[i])])
    
    pts = np.array(pts, np.int32)
    pts = pts.reshape((-1,1,2))
    vehicle_image = preprocess_image(image_path)
    
    cv2.polylines(vehicle_image,[pts],True,(0,255,0),thickness)
    return vehicle_image    