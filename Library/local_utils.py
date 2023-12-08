# pylint: disable=invalid-name, redefined-outer-name, missing-docstring, non-parent-init-called, trailing-whitespace, line-too-long
import cv2
import numpy as np
import matplotlib.pyplot as plt

class Label:
    def __init__(self, cl=-1, tl=np.array([0., 0.]), br=np.array([0., 0.]), prob=None):
        self.__tl = tl
        self.__br = br
        self.__cl = cl
        self.__prob = prob

    def __str__(self):
        return 'Class: %d, top left(x: %f, y: %f), bottom right(x: %f, y: %f)' % (
        self.__cl, self.__tl[0], self.__tl[1], self.__br[0], self.__br[1])

    def copy(self):
        return Label(self.__cl, self.__tl, self.__br)

    def wh(self): return self.__br - self.__tl

    def cc(self): return self.__tl + self.wh() / 2

    def tl(self): return self.__tl

    def br(self): return self.__br

    def tr(self): return np.array([self.__br[0], self.__tl[1]])

    def bl(self): return np.array([self.__tl[0], self.__br[1]])

    def cl(self): return self.__cl

    def area(self): return np.prod(self.wh())

    def prob(self): return self.__prob

    def set_class(self, cl):
        self.__cl = cl

    def set_tl(self, tl):
        self.__tl = tl

    def set_br(self, br):
        self.__br = br

    def set_wh(self, wh):
        cc = self.cc()
        self.__tl = cc - .5 * wh
        self.__br = cc + .5 * wh

    def set_prob(self, prob):
        self.__prob = prob

class DLabel(Label):
    def __init__(self, cl, pts, prob):
        self.pts = pts
        tl = np.amin(pts, axis=1)
        br = np.amax(pts, axis=1)
        Label.__init__(self, cl, tl, br, prob)

def getWH(shape):
    return np.array(shape[1::-1]).astype(float)

def IOU(tl1, br1, tl2, br2):
    wh1, wh2 = br1-tl1, br2-tl2
    assert((wh1 >= 0).all() and (wh2 >= 0).all())
    
    intersection_wh = np.maximum(np.minimum(br1, br2) - np.maximum(tl1, tl2), 0)
    intersection_area = np.prod(intersection_wh)
    area1, area2 = (np.prod(wh1), np.prod(wh2))
    union_area = area1 + area2 - intersection_area
    return intersection_area/union_area

def IOU_labels(l1, l2):
    return IOU(l1.tl(), l1.br(), l2.tl(), l2.br())

def nms(Labels, iou_threshold=0.5):
    SelectedLabels = []
    Labels.sort(key=lambda l: l.prob(), reverse=True)
    
    for label in Labels:
        non_overlap = True
        for sel_label in SelectedLabels:
            if IOU_labels(label, sel_label) > iou_threshold:
                non_overlap = False
                break

        if non_overlap:
            SelectedLabels.append(label)
    return SelectedLabels



def find_T_matrix(pts, t_pts):
    A = np.zeros((8, 9))
    for i in range(0, 4):
        xi = pts[:, i]
        xil = t_pts[:, i]
        xi = xi.T
        
        A[i*2, 3:6] = -xil[2]*xi
        A[i*2, 6:] = xil[1]*xi
        A[i*2+1, :3] = xil[2]*xi
        A[i*2+1, 6:] = -xil[0]*xi

    [U, S, V] = np.linalg.svd(A)
    H = V[-1, :].reshape((3, 3))
    return H

def getRectPts(tlx, tly, brx, bry):
    return np.matrix([[tlx, brx, brx, tlx], [tly, tly, bry, bry], [1, 1, 1, 1]], dtype=float)

def normal(pts, side, mn, MN):
    pts_MN_center_mn = pts * side
    pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
    pts_prop = pts_MN / MN.reshape((2, 1))
    return pts_prop

# Reconstruction function from predict value into plate crpoped from image 
#從預測值到從圖像轉換為印版的重構功能
def reconstruct(I, Iresized, Yr, lp_threshold):
    # 4 max-pooling layers, stride = 2      池化層
    net_stride = 2**4 #不動
    side = ((208 + 40)/2)/net_stride #不動

    # one line and two lines license plate size 一條線和兩條線的車牌尺寸
    one_line = (300, 200)
    two_lines = (300, 200)

    Probs = Yr[..., 0] #不動
    Affines = Yr[..., 2:] #不動

    xx, yy = np.where(Probs > lp_threshold) #不動
    # CNN input image size 
    WH = getWH(Iresized.shape) #不動
    # output feature map size  輸出要素圖大小
    MN = WH/net_stride  #不動

    vxx = vyy = 0.5 #alpha #不動
    base = lambda vx, vy: np.matrix([[-vx, -vy, 1], [vx, -vy, 1], [vx, vy, 1], [-vx, vy, 1]]).T #不動
    labels = []
    labels_frontal = []
#----------不動
    for i in range(len(xx)):
        x, y = xx[i], yy[i]
        affine = Affines[x, y]
        prob = Probs[x, y]

        mn = np.array([float(y) + 0.5, float(x) + 0.5])

        # affine transformation matrix 仿射變換矩陣output feature map size
        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0)
        A[1, 1] = max(A[1, 1], 0)
        # identity transformation #身分轉換 
        B = np.zeros((2, 3))
        B[0, 0] = max(A[0, 0], 0)
        B[1, 1] = max(A[1, 1], 0)

        pts = np.array(A*base(vxx, vyy))
        pts_frontal = np.array(B*base(vxx, vyy))
#---------不動		
        pts_prop = normal(pts, side, mn, MN)
        frontal = normal(pts_frontal, side, mn, MN)

        labels.append(DLabel(0, pts_prop, prob))
        labels_frontal.append(DLabel(0, frontal, prob))
        
    final_labels = nms(labels, 0.1) #不動
    final_labels_frontal = nms(labels_frontal, 0.1)

    #print(final_labels)
    #print(final_labels_fronta)
    #assert final_labels_frontal, "No License plate is founded!"
    
    # LP size and type
    try:
        out_size, lp_type = (two_lines, 2) if ((final_labels_frontal[0].wh()[0] / final_labels_frontal[0].wh()[1]) < 1.7) else (one_line, 1)
        #out_size, lp_type = (two_lines, 2) if ((final_labels[0].wh()[0] / final_labels[0].wh()[1]) < 1.7) else (one_line, 1)
		#--------不動	
        TLp = []
        Cor = []
        if len(final_labels):
            final_labels.sort(key=lambda x: x.prob(), reverse=True)
            for _, label in enumerate(final_labels):
                t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
                ptsh = np.concatenate((label.pts * getWH(I.shape).reshape((2, 1)), np.ones((1, 4))))
                H = find_T_matrix(ptsh, t_ptsh)
                Ilp = cv2.warpPerspective(I, H, out_size, borderValue=0)
                    
                TLp.append(Ilp)
                Cor.append(ptsh)
            
		#--------不動		
    except:
        return None,None,None,None
        pass 
        
    return final_labels, TLp, lp_type, Cor

	
	
def reconstruct2(Iorig,I,Yr,out_size,threshold=.9):
    # 4 max-pooling layers, stride = 2      池化層
    net_stride = 2**4 #不動 2的四次方
    side = ((208 + 40)/2)/net_stride #不動 7.75

    # one line and two lines license plate size 一條線和兩條線的車牌尺寸
    #one_line = (300, 200)
    #two_lines = (300, 200)

    Probs = Yr[..., 0] #不動  #模型的預測結果概率
    Affines = Yr[..., 2:] #不動 #仿設變換的
	
    rx, ry = Yr.shape[:2] #new
    ywh = Yr.shape[1::-1] #new
    iwh = np.array(I.shape[1::-1],dtype=float).reshape((2,1))
    xx, yy = np.where(Probs > threshold) #不動 
    # CNN input image size 
    WH = getWH(I.shape) #不動
    # output feature map size  輸出要素圖大小
    MN = WH/net_stride  #不動

    vxx = vyy = 0.5 #alpha #不動
    base = lambda vx, vy: np.matrix([[-vx, -vy, 1], [vx, -vy, 1], [vx, vy, 1], [-vx, vy, 1]]).T #不動 構建放射矩陣
    labels = []
    labels_frontal = []
#----------不動
    for i in range(len(xx)):
        x, y = xx[i], yy[i]
        affine = Affines[y, x]
        prob = Probs[y, x]

        mn = np.array([float(x) + 0.5, float(y) + 0.5])
        # affine transformation matrix 仿射變換矩陣output feature map size
        A = np.reshape(affine, (2, 3))
        A[0, 0] = max(A[0, 0], 0)
        A[1, 1] = max(A[1, 1], 0)
        # identity transformation #身分轉換 
        #B = np.zeros((2, 3))
        #B[0, 0] = max(A[0, 0], 0)
        #B[1, 1] = max(A[1, 1], 0)

        pts = np.array(A*base(vxx, vyy))
        #pts_frontal = np.array(B*base(vxx, vyy))
        #pts_MN_center_mn = pts*side
        #pts_MN = pts_MN_center_mn + mn.reshape((2,1))
		
#---------不動		
        pts_MN_center_mn = pts * side
        pts_MN = pts_MN_center_mn + mn.reshape((2, 1))
        pts_prop = pts_MN / MN.reshape((2, 1))
        #pts_prop = normal(pts, side, mn, MN)
        #frontal = normal(pts_frontal, side, mn, MN)

        labels.append(DLabel(0, pts_prop, prob))
        #labels_frontal.append(DLabel(0, frontal, prob))  
    final_labels = nms(labels, 0.1) #不動
    #final_labels_frontal = nms(labels_frontal, 0.1)

    #print(final_labels)
    #print(final_labels_fronta)
    #assert final_labels_frontal, "No License plate is founded!"
    
    # LP size and type
    #out_size, lp_type = (two_lines, 2)
    TLp = []
    Cor = []
    if len(final_labels):
         final_labels.sort(key=lambda x: x.prob(), reverse=True)
         for i,label in enumerate(final_labels):

            t_ptsh 	= getRectPts(0,0,out_size[0],out_size[1])
            ptsh 	= np.concatenate((label.pts*getWH(Iorig.shape).reshape((2,1)),np.ones((1,4))))
            H 		= find_T_matrix(ptsh,t_ptsh)
            Ilp 	= cv2.warpPerspective(Iorig,H,out_size,borderValue=.0)
            Cor.append(ptsh)
            TLp.append(Ilp)	
#	try:
#        out_size, lp_type = (two_lines, 2) if ((final_labels_frontal[0].wh()[0] / final_labels_frontal[0].wh()[1]) < 1.7) else (one_line, 1)
        #out_size, lp_type = (two_lines, 2) if ((final_labels[0].wh()[0] / final_labels[0].wh()[1]) < 1.7) else (one_line, 1)
		#--------不動	
#        TLp = []
#        Cor = []
#        if len(final_labels):
#            final_labels.sort(key=lambda x: x.prob(), reverse=True)
#            for _, label in enumerate(final_labels):
#                t_ptsh = getRectPts(0, 0, out_size[0], out_size[1])
#                ptsh = np.concatenate((label.pts * getWH(I.shape).reshape((2, 1)), np.ones((1, 4))))
#                H = find_T_matrix(ptsh, t_ptsh)
#               #print(out_size[0], out_size[1])
#                Ilp = cv2.warpPerspective(I, H, out_size, borderValue=0)
#                TLp.append(Ilp)
#                Cor.append(ptsh)
            
		#--------不動		
#    except:
#        return None,None,None,None
#        pass   
    return final_labels, TLp, Cor	
	
	
	
	
	
	
	
	
	
	
	
	
	
def detect_lp(model, I, max_dim, lp_threshold):
    min_dim_img = min(I.shape[:2])
    factor = float(max_dim) / min_dim_img #計算縮放因子
    w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist() #根據縮放因子和車輛圖像計算w和h
    #---------
    #net_step = 2**4
    #w += (w%net_step!=0)*(net_step - w%net_step)
    #h += (h%net_step!=0)*(net_step - h%net_step)    
    #-------
    Iresized = cv2.resize(I, (w, h))  #車輛圖片縮放
    plt.imshow(Iresized)
    T = Iresized.copy()
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
    Yr = model.predict(T) #預測車牌區域
    Yr = np.squeeze(Yr) #刪維度 只對其有1維度作用
    L, TLp, lp_type, Cor = reconstruct(I, Iresized, Yr, lp_threshold)
    return L, TLp, lp_type, Cor


def detect_lp2(model, I, max_dim, lp_threshold):
    out_size = (240,80)
    min_dim_img = min(I.shape[:2]) # [W,H,C]-->[W,H] 提取前面兩個數
    factor = float(max_dim) / min_dim_img 
    
    w, h = (np.array(I.shape[1::-1], dtype=float) * factor).astype(int).tolist() # [w,h,c]
    #---------
    net_step = 2**4
    w += (w%net_step!=0)*(net_step - w%net_step)
    h += (h%net_step!=0)*(net_step - h%net_step)    
    Iresized = cv2.resize(I, (w, h)) #車輛圖片縮放
    #--------- 
    plt.imshow(Iresized)
    #---------
    
    T = Iresized.copy()
    T = T.reshape((1, T.shape[0], T.shape[1], T.shape[2]))
    

    Yr = model.predict(T) #預測車牌區域
    print(Yr)
    Yr = np.squeeze(Yr) #刪維度 只對其有1維度作用
    
    L, TLp, Cor = reconstruct2(I, Iresized, Yr, out_size, lp_threshold) #放射變換 矯正車牌
    return L, TLp, Cor

 
    