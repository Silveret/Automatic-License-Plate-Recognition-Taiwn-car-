# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'predict.ui'
#
# Created by: PyQt5 UI code generator 5.9.2
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets
import tkinter as tk
from tkinter import filedialog
from PyQt5.QtGui import QPixmap
import cv2
from UI import image_path

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(500, 500)
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(40, 10, 300, 300))
        self.graphicsView.setObjectName("graphicsView")
        self.label = QtWidgets.QLabel(Dialog)
        self.label.setGeometry(QtCore.QRect(100, 320, 141, 50))
        self.label.setObjectName("label")
        self.label2 = QtWidgets.QLabel(Dialog)
        self.label2.setGeometry(QtCore.QRect(150, 360, 141, 50))
        self.label2.setObjectName("Predict")
        self.pushButton = QtWidgets.QPushButton(Dialog)
        self.pushButton.setGeometry(QtCore.QRect(170, 320, 141, 50))    
        self.pushButton.setObjectName("pushButton")
        self.pushButton.clicked.connect(self.Action)
        
        self.graphicsScene = QtWidgets.QGraphicsScene(Dialog)

        
        self.retranslateUi(Dialog)
    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.label.setText(_translate("Dialog", "匯入圖片") )
        self.pushButton.setText(_translate("Dialog", "選擇圖片")) 
        self.label2.setText(_translate("Dialog", "這邊顯示辨識結果") )
    def Action(self):
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        if file_path:
            pix = QPixmap(file_path)
            item = QtWidgets.QGraphicsPixmapItem(pix)
            self.graphicsScene.addItem(item)
            self.graphicsView.setScene(self.graphicsScene)
            string = image_path(file_path)  
            self.label2.setText("辨識結果為" + string) 
               






        