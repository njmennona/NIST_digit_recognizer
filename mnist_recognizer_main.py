#%%
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
import cv2
from matplotlib import pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
from digitRecognizer_Kaggle.Network import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
#%%
file_EXT = 'train.csv'
dirName = Path('C:/Users/nicho/Documents/KAGGLE')  # Using forward slashes or pathlib Path
foldName = 'digitRecognizer_Kaggle'
fileName = dirName/foldName / file_EXT
nistFile = pd.read_csv(fileName)
# %% CREATE MATRIX OF IMAGES
labels = []
ims = []
imSize = np.int(np.sqrt(nistFile.shape[1]-1))
for i,r in nistFile.iterrows():
    labels.append(r['label'])
    # ims.append(r[:,1:])
    ims.append(r[1:].values.reshape((imSize,imSize)).astype(float)/255) #r is a series, so i need to do r.values to get the
    # ndarray
    # ims.append(r[:,2:].reshape((28,28)))
# imSize = np.int(np.sqrt(nistIms.shape[1]))
# numIms = nistIms.shape[0]
# nist_Data = np.ndarray((imSize,imSize,numIms))
# for imIdx in range(numIms):
#     for i in range(imSize-1):
#         for j in range(imSize-1):
#             idx = i*28+j
#             nist_Data[i,j,imIdx]=nistIms.iloc[imIdx,idx]

# plt.imshow(ims[0])
# %% Save the data into png and txt file
# %% Train model
model = Model((28,28,1),10)
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
batch_size = 32
num_epochs=10
images = np.array(ims)
images_train = np.expand_dims(images, axis=-1) 
labelArray = np.array(labels)
label_Model = tf.keras.utils.to_categorical(labelArray.reshape(-1, 1), 10)
model.fit(images_train,label_Model,batch_size=32,epochs=1)
# %% PREPROCESS/FILTER DATA
# blur = cv.GaussianBlur(img,(5,5),0) #need to see if augmenting the data can improve the performance
# model.summary()
# %% Test data
test_EXT='test.csv'
testFile = dirName/foldName/test_EXT
testFile_ims = pd.read_csv(testFile)
imTEST = []
imSize = np.int(np.sqrt(testFile_ims.shape[1]))
for i,r in testFile_ims.iterrows():
    imTEST.append(r.values.reshape((imSize,imSize)).astype(float)/255)
images_TEST = np.array(imTEST)
images_test = np.expand_dims(images_TEST, axis=-1)
#%% 
test_pred = model.predict(images_test)
output = pd.DataFrame(
    data = pd.DataFrame(test_pred).apply(lambda x: x.index[x == max(x)].values[0], axis=1).reset_index().values,
    columns = ['ImageId', 'Label']
)
output['ImageId'] += 1
output.to_csv(r'C:/Users/nicho/Documents/KAGGLE/digitRecognizer_Kaggle/submission.csv', index=False)
# %%
