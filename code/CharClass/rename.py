import os

list1 = ["A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z","0","1","2","3","4","5","6","7","8","9"]
for x in range(len(list1)):
    os.system("cd " +list1[x]+"&&rename *.jpg *.png")
    os.system("cd ..")