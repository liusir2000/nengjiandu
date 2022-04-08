import cv2
import numpy as np
import os
import sys
import datetime


def getImageAvg(imgPath):
    image = cv2.imread(imgPath);
    img2gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    imageVar = cv2.Laplacian(img2gray, cv2.CV_64F)
    return imageVar

def getTidu(filename):
    img = cv2.imread(filename, 0)    
    sobelx=cv2.Sobel(img,cv2.CV_64F,dx=1,dy=0)
    sobelx=cv2.convertScaleAbs(sobelx)
    sobely=cv2.Sobel(img,cv2.CV_64F,dx=0,dy=1)
    sobely=cv2.convertScaleAbs(sobely)
    result=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    return result

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
    #else:
    #    if(img.shape[0]>1080):
    #        img = img[700:,0:1920]    
   
    HSV_img = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    x = HSV_img.T
    

    sobelx=cv2.Sobel(img,cv2.CV_64F,dx=1,dy=0)
    sobelx=cv2.convertScaleAbs(sobelx)
    sobely=cv2.Sobel(img,cv2.CV_64F,dx=0,dy=1)
    sobely=cv2.convertScaleAbs(sobely)
    result=cv2.addWeighted(sobelx,0.5,sobely,0.5,0)
    tidufile = 'tidu.png'                
    cv2.imwrite(tidufile, result)
                    
    #outputfile= outDir + filename
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
    #cv2.imwrite(outputfile, r)
    filename = os.path.split(filename)[-1]
    outputfile= os.path.join(outDir,filename.split('.')[0]+'_'+timestr+'.png')	
    rgb=cv2.merge([b,g,r])
    cv2.imwrite(outputfile, rgb)
    return outputfile

#cv2.imshow('HSV format image0',x[0].T)
#cv2.imshow('HSV format image1',x[1].T)
#cv2.imshow('HSV format image2',x[2].T)
#
#while(True): 
#	if cv2.waitKey(30) & 0xff == ord('q'):
#		break
#cv2.destroyAllWindows() 


        
if __name__=='__main__':
    theDir = '.\\uploadmap\\'
    outDir = r'.\\TrainSet\\' 
    print(len(sys.argv))
    timestr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')    
    try:
        if(len(sys.argv)==3):
            filename = sys.argv[1]           
            subDir = sys.argv[2]
            newDir = os.path.join(outDir,subDir)
            
            if not os.path.exists(newDir):
                os.makedirs(newDir)
                
            afilename = os.path.join(theDir,filename)
            sourcefile = dealtoRgb(afilename,newDir)
    
    
            enlargedir = newDir
             
            img = cv2.imread(sourcefile, cv2.IMREAD_COLOR)
            x = img.T
            b = x[0].T
            j = 0
            for i in range(b.shape[0]-1):
                if((j%5)!=0):
                    j = j + 1
                    continue            
                b = x[0].T.copy()
                n = b[i]
                b[i+1] = b[i]
                b[i] = n
                
                g = x[1].T.copy()
                n = g[i]
                g[i+1] = g[i]
                g[i] = n
                
                r = x[2].T.copy()
                n = r[i]
                r[i+1] = r[i]
                r[i] = n    
                
                outputfile = filename.split('.')[0]+'_'+timestr+'_'+str(j)+'.png'
                outputfile = os.path.join(enlargedir,outputfile)
                rgb=cv2.merge([b,g,r])
                cv2.imwrite(outputfile, rgb)
                j = j + 1
            print('ok')
        else:
            print('error 1',sys.argv)
    except Exception as err:
        print('error 0',err)



















    
       
        
