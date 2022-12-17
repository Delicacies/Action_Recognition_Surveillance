# 第2步：生成图片路径train.txt

import os
import random 
 
# xmlfilepath=r"./labels_old/"
txtfilepath=r"./celltosmoke/txt/txt_test"
imgsPath = './test'
txt_name = 'test.txt'
saveBasePath=r"./celltosmoke/"
 
trainval_percent=1
train_percent=0.9
total_xml = os.listdir(txtfilepath)
num=len(total_xml)  
list=range(num)  
'''
tv=int(num*trainval_percent)  
tr=int(tv*train_percent)  
trainval= random.sample(list,tv)  
train=random.sample(trainval,tr)   
print("train and val size",tv)
print("traub suze",tr)
'''

#ftrainval = open(os.path.join(saveBasePath,'trainval.txt'), 'w')  
#ftest = open(os.path.join(saveBasePath,'test.txt'), 'w')  
ftrain = open(os.path.join(saveBasePath,txt_name), 'w')  
# fval = open(os.path.join(saveBasePath,'valid.txt'), 'w')  
 
for i  in list:  
    name=os.path.join(imgsPath, total_xml[i][:-4] + ".jpg") +'\n'  
    ftrain.write(name)
    '''
    if i in trainval:  
        #ftrainval.write(name)  
        if i in train:  
            ftrain.write(name)  
        else:  
            fval.write(name)  
    else:  
        #ftest.write(name)
        print("==")
    '''
#ftrainval.close()
ftrain.close()
# fval.close()
#ftest .close()

print("{} has been written to {} !!".format(txt_name, saveBasePath+txt_name))
