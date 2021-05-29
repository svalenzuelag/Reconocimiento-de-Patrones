import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2 as cv

img = cv.imread('mis_numeros/num_5.jpg')
img_Ga = cv.GaussianBlur(img,(7,7),0)
img_g = cv.cvtColor(img_Ga, cv.COLOR_BGR2GRAY)
img_r = cv.resize(img_g,(28,28), interpolation=cv.INTER_NEAREST)
img_i = cv.bitwise_not(img_r)
thr, img_f= cv.threshold(img_i,135,255, cv.THRESH_TOZERO)
#plt.imshow(img_f, cmap='gray')
#plt.show()

plt.figure(figsize=(10,10))
plt.subplot(2,3,1)
plt.imshow(img)
plt.subplot(2,3,2)
plt.imshow(img_Ga)
plt.subplot(2,3,3)
plt.imshow(img_g, cmap='gray')
plt.subplot(2,3,4)
plt.imshow(img_r, cmap='gray')
plt.subplot(2,3,5)
plt.imshow(img_i, cmap='gray')
plt.subplot(2,3,6)
plt.imshow(img_f, cmap='gray')
plt.show()

plt.figure(figsize=(8,8))
plt.imshow(img_f, cmap='gray')
for (j,i), label in np.ndenumerate(img_f):
    plt.text(i,j, label, ha='center', va='center', color='red')
    #plt.show

def digits (img):
    img_Ga = cv.GaussianBlur(img,(7,7),0)
    img_g = cv.cvtColor(img_Ga, cv.COLOR_BGR2GRAY)
    img_r = cv.resize(img_g,(28,28), interpolation=cv.INTER_NEAREST)
    img_i = cv.bitwise_not(img_r)
    thr, img_f= cv.threshold(img_i,135,255, cv.THRESH_TOZERO)
    return (img_f)

from glob import glob
contador=0
df= pd.DataFrame()
for fn in glob('mis_numeros/num_*.jpg'):
    print(contador)
    img_f=digits(cv.imread(fn))
    plt.imshow(img_f, cmap='gray')
    plt.show()
    fila=np.append(contador, img_f.flatten())
    fila = fila.reshape(1,785)
    df = df.append(pd.DataFrame(fila))
    contador = contador+1

    df.to_csv('mis_numeros.csv', header=False, index=False)