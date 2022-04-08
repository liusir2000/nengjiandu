from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from keras.models import load_model
import os
from keras.preprocessing import image
import cv2
from astral.sun import sun

modelname = 'rebuildModel.h5'
xsize = 256
ysize = 256

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def getTidu(filename):
    img = cv2.imread(filename, 0)    
    sobelx=cv2.Sobel(img,cv2.CV_64F,dx=1,dy=0)
    sobelx=cv2.convertScaleAbs(sobelx)
    sobely=cv2.Sobel(img,cv2.CV_64F,dx=0,dy=1)
    sobely=cv2.convertScaleAbs(sobely)
    result=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    return np.mean(result)


#获得平均饱和度和亮度
def getAvgSAndV(img):
    HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    x = HSV_img.T
    yxg=x[1]
    yyg=yxg.T
    vvg = np.asarray(yyg) 
    yxb=x[2]
    yyb=yxb.T
    vvb = np.asarray(yyb)      
    return np.mean(vvg),np.mean(vvb)

def countHist(img):
    result = np.zeros([256])
    for i in img:
        if i>=256:
            result[255] = result[255] + 1
        else:
            result[i] = result[i] + 1
    return result 


def dealtoRgb(filename,outDir):
    sourcefile= filename
    img = cv2.imread(sourcefile, cv2.IMREAD_COLOR)
    #print(img.shape)
    if(img.shape==(1080,1920,3)):
        img = img[150:1050,0:1920]
    
   
    
    HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    x = HSV_img.T
    
    #else:
    #    if(img.shape[0]>1080):
    #        img = img[700:,0:1920]
    sobelx=cv2.Sobel(img,cv2.CV_64F,dx=1,dy=0)
    sobelx=cv2.convertScaleAbs(sobelx)
    sobely=cv2.Sobel(img,cv2.CV_64F,dx=0,dy=1)
    sobely=cv2.convertScaleAbs(sobely)
    result=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    tidufile = os.path.join(outDir,'tiduimg.png')                
    cv2.imwrite(tidufile, result)
                    
    if not os.path.exists(outDir):
        os.makedirs(outDir)   
    img = cv2.imread(tidufile, 0)
    
    vvr = np.asarray(img)
    yxg=x[1]
    yyg=yxg.T
    vvg = np.asarray(yyg)
    
    yxb=x[2]
    yyb=yxb.T
    vvb = np.asarray(yyb)  
    
    r = np.empty([256,256], dtype=float)
    g = np.empty([256,256], dtype=float)
    b = np.empty([256,256], dtype=float)
    j = 0
    stepi = img.shape[0]/256
    i1 = 0
    while j<256:
        i1 = int(np.round(j*stepi))
        i2 = int(np.round((j+1)*stepi))
        vvr1 = np.reshape(vvr[i1:i2],(vvr[i1:i2].shape[0]*vvr[i1:i2].shape[1],))
        vvg1 = np.reshape(vvg[i1:i2],(vvg[i1:i2].shape[0]*vvg[i1:i2].shape[1],))
        vvb1 = np.reshape(vvb[i1:i2],(vvb[i1:i2].shape[0]*vvb[i1:i2].shape[1],))

        yr = countHist(vvr1)
        a = np.asarray(yr)
        r[j] = a
        
        yg = countHist(vvg1)
        a = np.asarray(yg)
        g[j] = a
        
        yb = countHist(vvb1)
        a = np.asarray(yb)
        b[j] = a        
        
        j = j + 1         
    
    outputfile= outDir + os.path.split(filename)[-1]
    rgb=cv2.merge([b,g,r])
    cv2.imwrite(outputfile, rgb)
    
#文件前处理
filename = 'tempnew.png'
sourcefilename = os.path.join('uploadmap\\',filename)
outDir =  'middelpic\\'
if not os.path.exists(outDir):
    os.makedirs(outDir)  
    

               
outputfile = os.path.join(outDir,filename)

dealtoRgb(sourcefilename,outDir)
 
images = []
imageNames = []
model = load_model(modelname)
image_size= (xsize,ysize)

train_idx = 0
npy_idx = 0
images = []
imageNames = []
labels = []
img_path = outputfile
img = image.load_img(img_path, target_size=image_size)
img_array = image.img_to_array(img)
images.append(img_array)
images = np.array(images)
preds = model.predict_classes(images,batch_size = 128)
for x in preds:
    print(x)

#np.savetxt("preds.txt",preds)
#np.savetxt("labels.txt",labels)
