# N. Mennona
# trying out some cnn architectures for the MNIST data
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer, Input
from tensorflow.keras.layers import Reshape, MaxPooling2D, Add, AveragePooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Conv2D, Dense, Flatten, BatchNormalization,Activation,Concatenate
import numpy as np

def Model(input_shape, num_classes):
    
    # SIMPLE CNN MODEL
    input_layer = Input(shape=input_shape)
    model = Sequential()
    model.add(Input(shape=(input_shape)))
    model.add(Conv2D(kernel_size=5, strides=1, filters=32, padding='same', activation='relu', name='conv1'))
    model.add(MaxPooling2D(pool_size=2, strides=2, name='maxpool1'))
    model.add(Conv2D(kernel_size=5, strides=1, filters=64, padding='same', activation='relu', name='conv2'))
    model.add(MaxPooling2D(pool_size=2, strides=2, name='maxpool2'))
    model.add(Flatten(name='flatten'))
    model.add(Dense(128, activation='relu', name='dense1'))
    model.add(Dense(num_classes, activation='softmax', name='output_layer'))
    return model


    # Implementing ResNet, ResXNet, and DenseNet architectures (Section 3.5)
    #for this model, we must use keras' functional interface
    # Try implementing an 18 layer ResNet (RESNET)
    # input_layer = Input(shape=input_shape)
    # x = Conv2D(filters=64,kernel_size=7,strides=2)(input_layer)
    # preRes_x = MaxPooling2D(pool_size=3,strides=2)(x)
    # x = Conv2D(filters=64,kernel_size=3,padding='same',activation='relu')(preRes_x)
    # x = Conv2D(filters=64,kernel_size=3,padding='same',activation='relu')(x)
    # newX=Add()([preRes_x,x])#ensures element-wise addition and matching dimensions
    # x = Conv2D(filters=64,kernel_size=3,strides=2,padding='same',activation='relu')(newX)
    # x = Conv2D(filters=64,kernel_size=3,padding='same',activation='relu')(x)
    # newX = Conv2D(filters=x.shape[-1], kernel_size=1,strides=2, padding='same')(newX)
    # newX=Add()([newX,x])
    # x = Conv2D(filters=128,kernel_size=3,strides=2,padding='same',activation='relu')(newX)
    # x = Conv2D(filters=128,kernel_size=3,padding='same',activation='relu')(x)
    # newX = Conv2D(filters=x.shape[-1], kernel_size=1,strides=2, padding='same')(newX)
    # # newX = [newX+x]
    # newX=Add()([newX,x])
    # x = Conv2D(filters=128,kernel_size=3,padding='same',activation='relu')(newX)
    # x = Conv2D(filters=128,kernel_size=3,padding='same',activation='relu')(x)
    # newX = Conv2D(filters=x.shape[-1], kernel_size=1, strides=2,padding='same')(newX)
    # # newX = [newX+x]
    # newX=Add()([newX,x])
    # x = Conv2D(filters=256,kernel_size=3,strides=2,padding='same',activation='relu')(newX)
    # x = Conv2D(filters=256,kernel_size=3,padding='same',activation='relu')(x)
    # newX = Conv2D(filters=x.shape[-1], kernel_size=1, strides=2,padding='same')(newX)
    # # newX = [newX+x]
    # newX=Add()([newX,x])
    # x = Conv2D(filters=256,kernel_size=3,padding='same',activation='relu')(newX)
    # x = Conv2D(filters=256,kernel_size=3,padding='same',activation='relu')(x)
    # newX = Conv2D(filters=x.shape[-1], kernel_size=1,strides=2, padding='same')(newX)
    # # newX = [newX+x]
    # newX=Add()([newX,x])
    # x = Conv2D(filters=512,kernel_size=3,strides=2,padding='same',activation='relu')(newX)
    # x = Conv2D(filters=512,kernel_size=3,padding='same',activation='relu')(x)
    # newX = Conv2D(filters=x.shape[-1], kernel_size=1, strides=2,padding='same')(newX)
    # # newX = [newX+x]
    # newX=Add()([newX,x])
    # x = Conv2D(filters=512,kernel_size=3,padding='same',activation='relu')(newX)
    # x = Conv2D(filters=512,kernel_size=3,padding='same',activation='relu')(x)
    # x = AveragePooling2D(pool_size=1,strides=1)(newX)
    # x=Flatten()(x)
    # x=Dense(1000,activation='relu')(x)
    # x=Dense(num_classes,activation='softmax',name='output_layer')(x)
    # # print(x.shape)
    # model = Model(inputs=input_layer,outputs=x)
    # return model

#         # RESXNET
#     input_layer = Input(shape=input_shape)
#     x = Conv2D(filters=64,kernel_size=7,strides=2)(input_layer)
#     preRes_x = MaxPooling2D(pool_size=3,strides=2)(x)
#     # cardinal=32
#     cardinal=4
#     numFilt = 32
#     #128 block
#     x=preRes_x
#     # numConLayers = np
#     # numConLayers = np.array([3,4,6,3])
#     # numConLayers = np.array([2,2])
#     numConLayers = np.array([2,2,2])
#     for ll in range(len(numConLayers)):
#         tempL = numConLayers[ll]
#         for ii in range(tempL):
#             if ii==1:
#                 x = residualX(x,numFilt*(ll+1),cardinal,True)#"downsampling of conv3,4,5 is done by stride-2 convolutions in the 3x3 layer of the first block"
#             else:
#                 x = residualX(x,numFilt*(ll+2),cardinal,False)
    
#     x = GlobalAveragePooling2D()(x)
#     x=Flatten()(x)
#     x=Dense(1000,activation='relu')(x)
#     x=Dense(10,activation='softmax',name='output_layer')(x)
#     # print(x.shape)
#     model = Model(inputs=input_layer,outputs=x)
#     return model














#     DENSENET
#     kSize = 12  # corresponds to the number of output feature vectors
#     numLayers = 6
#     input_layer = Input(shape=input_shape)

#     x = Conv2D(filters=kSize, kernel_size=3, strides=1, padding='same', activation='relu')(input_layer)#these params for CIFAR10, not ImageNet
#     x = MaxPooling2D(pool_size=(3, 3), strides=1)(x)
#     x = denseblock(x, kSize, numLayers)
#     x=transitionblock(x)
#     x = denseblock(x, kSize, numLayers)
#     x=transitionblock(x)
#     x = denseblock(x, kSize, numLayers)
#     x = GlobalAveragePooling2D()(x)
#     x = Flatten()(x)
#     x = Dense(1000, activation='relu')(x)
#     x = Dense(10, activation='softmax', name='output_layer')(x)
#     # print(x.shape)

#     model = Model(inputs=input_layer, outputs=x)
#     return model



# # def CIFAR10Model(input_shape, num_classes):
# def residual(initX,kernelSize,numKernels):
#     # modulate a residual block for constructing a multi-layer residual network
#     x = Conv2D(filters=numKernels,kernel_size=kernelSize,strides=2,padding='same',activation='relu')(initX)#first layer is always strides=2
#     x = Conv2D(filters=numKernels,kernel_size=kernelSize,padding='same',activation='relu')(x)
#     newX=Add()([initX,x])#ensures element-wise addition and matching dimensions

#     x = Conv2D(filters=numKernels,kernel_size=kernelSize,strides=2,padding='same',activation='relu')(newX)
#     x = Conv2D(filters=numKernels,kernel_size=kernelSize,padding='same',activation='relu')(x)
#     newX = Conv2D(filters=x.shape[-1], kernel_size=1,strides=2, padding='same')(newX)#necessary for each aggregate layer to ensure matched dimension sizes
#     newX=Add()([newX,x])
#     return newX
# def residualX(initX,initKSize,C,Downsample):
#     # C refers to the cardinality of the network
#     for cc in range(C):
#         if cc==0:
#             x=Conv2D(filters=initKSize,kernel_size=1,padding='same',activation='relu')(initX)
#             if Downsample:
#                 x = Conv2D(filters=initKSize,kernel_size=3,strides=2,padding='same',activation='relu')(x)
#             else:
#                 x = Conv2D(filters=initKSize,kernel_size=3,padding='same',activation='relu')(x)
#             x = Conv2D(filters=2*initKSize,kernel_size=1,padding='same',activation='relu')(x)
#             aggX=x
#         else:
#             x=Conv2D(filters=initKSize,kernel_size=1,padding='same',activation='relu')(initX)
#             if Downsample:
#                 x = Conv2D(filters=initKSize,kernel_size=3,strides=2,padding='same',activation='relu')(x)
#             else:
#                 x = Conv2D(filters=initKSize,kernel_size=3,padding='same',activation='relu')(x)
#             x = Conv2D(filters=2*initKSize,kernel_size=1,padding='same',activation='relu')(x)
#             aggX=Add()([aggX,x])
#     return aggX

# def denseblock(initX, k, numLayers):
#     allX = initX
#     for ii in range(2*numLayers):
#         kSize = 3 if np.mod(ii, 2) == 1 else 1
#         x = compositeH(allX, k, kSize)
#         allX = Concatenate(axis=-1)([allX, x])
#     return allX
# def transitionblock(denseX):
#     # x = BatchNormalization()(denseX)#adding this as definition in the text (not seen in Table 1)
#     x = Conv2D(filters=denseX.shape[-1], kernel_size=1, padding='same')(denseX)
#     x = AveragePooling2D(pool_size=2, strides=2)(x)
#     return x
# def compositeH(initX, k, kSize):
#     x = BatchNormalization()(initX)
#     x = Activation('relu')(x)
#     x = Conv2D(filters=k, kernel_size=kSize, padding='same')(x)#this function defines the H function as described in the paper
#     return x