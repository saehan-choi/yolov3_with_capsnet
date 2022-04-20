from email.headerregistry import HeaderRegistry
import pandas as pd
import shutil
import os

path = './PASCAL_VOC/'

list_images = os.listdir(path+'images_test')


# for i in list_images:
#     shutil.copy(path+'labels/'+i[:-4]+'.txt',path+'labels_test')

img_test = os.listdir(path+'images_test')
label_test = os.listdir(path+'labels_test')

df = pd.DataFrame()
df['img_test'] = img_test
df['label_test'] = label_test

df.to_csv(path+'train_test.csv',header=None, index=False)