# ML2021_FR_License_Plate_Recognition
車牌辨識(License plate recognition)
功能說明：
   
方法、及程式架構：

1.training
 
可視化數據集，數據集包含 34575 張圖像，分為 36 類字符 
 
數據預處理
排列輸入數據及其對應的標籤。 MobileNets 的輸入圖像的原始大小是 224x224，但是當嘗試堆疊輸入數據時 COLAB 不斷崩潰，所以將圖像大小減小到 80x80。此配置實現了 98% 的準確度 ，但是只要電腦足夠強大，仍然可以嘗試使用更大的圖像尺寸。
將我們的標籤作為一維數組轉換為單熱編碼標籤。 這使我們在標籤中具有更好的代表性。 標籤類記錄需要保存在本地，因此可以用於後面的逆變換。
將我們的數據集拆分為訓練集 (90%) 和驗證集 (10%)。 這使我們能夠監控模型的準確性並避免過度擬合。
使用一些基本的變換技術（如旋轉、移位、縮放等）創建數據增強方法。 注意不要過度使用這些技術，因為垂直翻轉的數字“6”會給我們數字“9”。
 
我們使用 imagenet 數據集上的預訓練權重構建我們的 MobileNets 。 該模型可以直接從 Keras 導入。
丟棄預設 MobileNets 架構的最後一個輸出層，並將其替換為我們想要的輸出層。 輸出層將包含與 34 個字符所關聯的 34 個節點。我們輸入圖像的形狀是 (80,80,3)，需要將我們的輸入層配置為相同的維度如果training=True，則將base_model中的所有層定義為可訓練層（權重可以在訓練過程中更新）； 初始化學習率和衰減值； 並使用損失和度量分別作為分類交叉熵和準確性來編譯我們的模型 
訓練模型
 
可視化訓練結果
模型架構需要保存，以便日後不必從頭開始構建我們的模型。 
 
將我們的正確率遺失率畫成圖表並儲存
 
參考: https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-3-recognize-be2eca1a9f12
 
plot可參考：https://www.southampton.ac.uk/~feeg1001/notebooks/Matplotlib.html
2.Predict
 
第一步：處理圖片
讀取圖片後將BGR轉RGB並縮放，會這麼做是因為cv2讀取圖片時是BGR而在Matplotlib顯示圖片會有色差問題，所以使用此方法可以避免發生此狀況
 
此函數會打包字元輪廓並排序字元輪廓尋找框座標超過偏差值，將錯誤輪廓剔除，由於在測試時因為台灣車牌長寬比沒有比國外大，所以無法只靠台灣長寬比將錯誤輪框框除，需要使用Z分數將錯誤輪廓踢除 
 
先使用get_plate 抓圖片中的車牌位置，會將車牌剪裁後進行轉正並傳回，之後將圖片進行處理
 
將矯正過後的車牌圖片轉成灰階在將圖片平滑且模糊化再來將圖片由黑轉白之後進行膨脹演算法
第二步：分割車牌上的字
 
將綠框的Y座標全部蒐集後計算其Z分數將超過1.5的值的綠框刪除與將長寬比大於7.5或低於1.5的綠框挑除之後將長小於車牌一半的綠框挑除
 
創建sort_contours（）函數以從左到右抓取每個數字的輪廓
 
迴圈所有的字元圖片，將每個字元使用函數進行預測，之後將預測的字元轉型成文字，並刪除不必要的符號寫進圖片
 
第三步：加載預訓練的MobileNets模型並進行預測
處理圖片參考：https://www.cnblogs.com/greentomlee/p/10863363.html
程式所使用到的套件，及開發執行環境說明：
cudnn-11.1-windows-x64-v8.0.4.30
cuda_11.1.0_456.43_win10
python3.8
Tensorflow-gpu =2.4.1
Keras=2.4.3
我們有匯出yaml檔可以直接設置，但是是使用cpu

 
環境設置可參考
https://blog.csdn.net/weixin_42101177/article/details/113512010
tensorflow-gpu運行tf.test.is_gpu_available()為false請參考(為ture就直接跑程式)
https://blog.csdn.net/chinakq/article/details/116354950?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-4&spm=1001.2101.3001.4242
      
資料來源：
https://www.cnblogs.com/greentomlee/p/10863363.html
https://blog.csdn.net/weixin_42101177/article/details/113512010
https://blog.csdn.net/chinakq/article/details/116354950?utm_medium=distribute.pc_relevant.none-task-blog-baidujs_title-4&spm=1001.2101.3001.4242
https://www.southampton.ac.uk/~feeg1001/notebooks/Matplotlib.html
https://www.itread01.com/hkepl.html
https://medium.com/@quangnhatnguyenle/detect-and-recognize-vehicles-license-plate-with-machine-learning-and-python-part-3-recognize-be2eca1a9f12




