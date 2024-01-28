#Ref. https://github.com/avivga/audio-visual-speech-enhancement
#https://machinelearningmastery.com/how-to-implement-major-architecture-innovations-for-convolutional-neural-networks/

from keras import optimizers
from keras.layers import Input, MaxPooling3D,  Conv3D,  GlobalAveragePooling3D
from keras.layers import MaxPooling2D, Conv2D,  GlobalAveragePooling2D
from keras.layers import  BatchNormalization, ReLU, ReLU,  Concatenate, Dense
from keras.layers.merge import concatenate, add
from keras.models import Model

#from keras.layers.convolutional import Deconv3D, Deconv2D, ZeroPadding3D

def video_net():
    
    video_shape = (128, 128, 64, 3)
    video_input = Input(shape=video_shape)
    nb_neurons = 16
    
    #******************************************** START Video BRANCH *******************************************************
    # BLOCK 1: Video Main
    conv_x1v = Conv3D(filters=nb_neurons, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(video_input)
    conc1xv = Concatenate(axis=4)([video_input, conv_x1v])   
    conv_y1v = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc1xv)
    conc1yv = Concatenate(axis=4)([video_input, conv_y1v])
    conv_z1v = Conv3D(filters=nb_neurons, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc1yv)
    conc1zv = Concatenate(axis=4)([video_input, conv_z1v])
    mp_z1v = MaxPooling3D(pool_size=(2,2, 2), padding='same')(conc1zv)
    st_y1v = Conv3D(filters=nb_neurons, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(video_input)
    conc1stv = Concatenate(axis=4)([video_input, st_y1v]) 
    mp_st1v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc1stv)
    out_1v = add([mp_st1v, mp_z1v])
    out_c1v = Conv3D(filters=nb_neurons*2, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_1v)
    out_c1vx = BatchNormalization()(out_c1v)
    
    # BLOCK 10: Video 1 RESIDUAL 01
    conv_x10rv = Conv3D(filters=nb_neurons, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(video_input)
    conc1x0rv = Concatenate(axis=4)([video_input, conv_x10rv])   
    conv_y10rv = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc1x0rv)
    conc1y0rv = Concatenate(axis=4)([video_input, conv_y10rv])
    conv_z10rv = Conv3D(filters=nb_neurons, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc1y0rv)
    conc1z0rv = Concatenate(axis=4)([video_input, conv_z10rv])
    mp_z10rv = MaxPooling3D(pool_size=(4,4, 4), padding='same')(conc1z0rv)
    st_y10rv = Conv3D(filters=nb_neurons, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(video_input)
    conc10rstv = Concatenate(axis=4)([video_input, st_y10rv]) 
    mp_st10rv = MaxPooling3D(pool_size=(4,4,4), padding='same')(conc10rstv)
    out_10rv = add([mp_st10rv, mp_z10rv])
    out_c10rv = Conv3D(filters=nb_neurons*4, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_10rv)
    out_c10rvx = BatchNormalization()(out_c10rv)
    
    # BLOCK 2:Video Main
    conv_x2v = Conv3D(filters=nb_neurons*2, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1vx)
    conc2xv = Concatenate(axis=4)([out_c1vx, conv_x2v])
    conv_y2v = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc2xv)
    conc2yv = Concatenate(axis=4)([out_c1vx, conv_y2v])
    conv_z2v = Conv3D(filters=nb_neurons*2, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc2yv)
    conc2zv = Concatenate(axis=4)([out_c1vx, conv_z2v])
    mp_z2v = MaxPooling3D(pool_size=(2, 2, 2), padding='same')(conc2zv)
    st_y2v = Conv3D(filters=nb_neurons*2, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1vx)
    conc2stv = Concatenate(axis=4)([out_c1vx, st_y2v])
    mp_st2v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc2stv)
    #out_2v = add([mp_st2v, mp_z2v])
    out_2v = add([mp_st2v, mp_z2v, out_c10rvx])
    out_c2v = Conv3D(filters=nb_neurons*4, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_2v)
    out_c2vx = BatchNormalization()(out_c2v)
    
    
    # BLOCK 2: Video 2 RESIDUAL 00
    conv_x20rv = Conv3D(filters=nb_neurons, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(video_input)
    conc2x0rv = Concatenate(axis=4)([video_input, conv_x20rv])   
    conv_y20rv = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc2x0rv)
    conc2y0rv = Concatenate(axis=4)([video_input, conv_y20rv])
    conv_z20rv = Conv3D(filters=nb_neurons, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc2y0rv)
    conc2z0rv = Concatenate(axis=4)([video_input, conv_z20rv])
    mp_z20rv = MaxPooling3D(pool_size=(8,8, 8), padding='same')(conc2z0rv)
    st_y20rv = Conv3D(filters=nb_neurons, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(video_input)
    conc20rstv = Concatenate(axis=4)([video_input, st_y20rv]) 
    mp_st20rv = MaxPooling3D(pool_size=(8,8,8), padding='same')(conc20rstv)
    out_20rv = add([mp_st20rv, mp_z20rv])
    out_c20rv = Conv3D(filters=nb_neurons*8, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_20rv)
    out_c20rvx = BatchNormalization()(out_c20rv)
    
    
    # BLOCK 2: Video 2 RESIDUAL 01
    conv_x21rv = Conv3D(filters=nb_neurons*2, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1vx)
    conc21rxv = Concatenate(axis=4)([out_c1vx, conv_x21rv])
    conv_y21rv = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc21rxv)
    conc21ryv = Concatenate(axis=4)([out_c1vx, conv_y21rv])
    conv_z21rv = Conv3D(filters=nb_neurons*2, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc21ryv)
    conc21rzv = Concatenate(axis=4)([out_c1vx, conv_z21rv])
    mp_z21rv = MaxPooling3D(pool_size=(4, 4, 4), padding='same')(conc21rzv)
    st_y21rv = Conv3D(filters=nb_neurons*2, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1vx)
    conc21rstv = Concatenate(axis=4)([out_c1vx, st_y21rv])
    mp_st21rv = MaxPooling3D(pool_size=(4,4,4), padding='same')(conc21rstv)
    out_21rv = add([mp_st21rv, mp_z21rv])
    out_c21rv = Conv3D(filters=nb_neurons*8, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_21rv)
    out_c21rvx = BatchNormalization()(out_c21rv)
    
             
    # BLOCK 3: Video Main
    conv_x3v = Conv3D(filters=nb_neurons*4, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2vx)
    conc3xv = Concatenate(axis=4)([out_c2vx, conv_x3v])
    conv_y3v = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc3xv)
    conc3yv = Concatenate(axis=4)([out_c2vx, conv_y3v])
    conv_z3v = Conv3D(filters=nb_neurons*4, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc3yv)
    conc3zv = Concatenate(axis=4)([out_c2vx, conv_z3v])
    mp_z3v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc3zv)
    st_y3v = Conv3D(filters=nb_neurons*4, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2vx)
    conc3stv = Concatenate(axis=4)([out_c2vx, st_y3v])
    mp_st3v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc3stv)
    #out_3v = add([mp_st3v, mp_z3v])
    out_3v = add([mp_st3v, mp_z3v, out_c20rvx, out_c21rvx])
    out_c3v = Conv3D(filters=nb_neurons*8, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_3v)
    out_c3vx = BatchNormalization()(out_c3v)
    
    # BLOCK 3: Video 3 RESIDUAL 00
    conv_x30rv = Conv3D(filters=nb_neurons, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(video_input)
    conc3x0rv = Concatenate(axis=4)([video_input, conv_x30rv])   
    conv_y30rv = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc3x0rv)
    conc3y0rv = Concatenate(axis=4)([video_input, conv_y30rv])
    conv_z30rv = Conv3D(filters=nb_neurons, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc3y0rv)
    conc3z0rv = Concatenate(axis=4)([video_input, conv_z30rv])
    mp_z30rv = MaxPooling3D(pool_size=(16,16, 16), padding='same')(conc3z0rv)
    st_y30rv = Conv3D(filters=nb_neurons, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(video_input)
    conc30rstv = Concatenate(axis=4)([video_input, st_y30rv]) 
    mp_st30rv = MaxPooling3D(pool_size=(16,16,16), padding='same')(conc30rstv)
    out_30rv = add([mp_st30rv, mp_z30rv])
    out_c30rv = Conv3D(filters=nb_neurons*16, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_30rv)
    out_c30rvx = BatchNormalization()(out_c30rv)
    
    
    # BLOCK 3: Video 3 RESIDUAL 01
    conv_x31rv = Conv3D(filters=nb_neurons*2, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1vx)
    conc31rxv = Concatenate(axis=4)([out_c1vx, conv_x31rv])
    conv_y31rv = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc31rxv)
    conc31ryv = Concatenate(axis=4)([out_c1vx, conv_y31rv])
    conv_z31rv = Conv3D(filters=nb_neurons*2, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc31ryv)
    conc31rzv = Concatenate(axis=4)([out_c1vx, conv_z31rv])
    mp_z31rv = MaxPooling3D(pool_size=(8, 8, 8), padding='same')(conc31rzv)
    st_y31rv = Conv3D(filters=nb_neurons*2, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1vx)
    conc31rstv = Concatenate(axis=4)([out_c1vx, st_y31rv])
    mp_st31rv = MaxPooling3D(pool_size=(8,8,8), padding='same')(conc31rstv)
    out_31rv = add([mp_st31rv, mp_z31rv])
    out_c31rv = Conv3D(filters=nb_neurons*16, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_31rv)
    out_c31rvx = BatchNormalization()(out_c31rv)
    
    # BLOCK 3: Video 3 RESIDUAL 02
    conv_x32rv = Conv3D(filters=nb_neurons*2, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2vx)
    conc32rxv = Concatenate(axis=4)([out_c2vx, conv_x32rv])
    conv_y32rv = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc32rxv)
    conc32ryv = Concatenate(axis=4)([out_c2vx, conv_y32rv])
    conv_z32rv = Conv3D(filters=nb_neurons*2, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc32ryv)
    conc32rzv = Concatenate(axis=4)([out_c2vx, conv_z32rv])
    mp_z32rv = MaxPooling3D(pool_size=(4, 4, 4), padding='same')(conc32rzv)
    st_y32rv = Conv3D(filters=nb_neurons*2, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2vx)
    conc32rstv = Concatenate(axis=4)([out_c2vx, st_y32rv])
    mp_st32rv = MaxPooling3D(pool_size=(4,4,4), padding='same')(conc32rstv)
    out_32rv = add([mp_st32rv, mp_z32rv])
    out_c32rv = Conv3D(filters=nb_neurons*16, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_32rv)
    out_c32rvx = BatchNormalization()(out_c32rv)
    
    
    # BLOCK 4: Video Main
    conv_x4v = Conv3D(filters=nb_neurons*8, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_c3vx)
    conc4xv = Concatenate(axis=4)([out_c3vx, conv_x4v])
    conv_y4v = Conv3D(filters=1, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc4xv)
    conc4yv = Concatenate(axis=4)([out_c3vx, conv_y4v])
    conv_z4v = Conv3D(filters=nb_neurons*8, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc4yv)
    conc4zv = Concatenate(axis=4)([out_c3vx, conv_z4v])
    mp_z4v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc4zv)
    st_y4v = Conv3D(filters=nb_neurons*8, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c3vx)
    conc4stv = Concatenate(axis=4)([out_c3vx, st_y4v])
    mp_st4v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc4stv)
    #out_4v = add([mp_st4v, mp_z4v])
    out_4v = add([mp_st4v, mp_z4v, out_c30rvx, out_c31rvx, out_c32rvx])
    out_c4v = Conv3D(filters=nb_neurons*16, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_4v)
    out_c4vx = BatchNormalization()(out_c4v)
    
    # BLOCK 4: Video 4 RESIDUAL 00
    conv_x40rv = Conv3D(filters=nb_neurons, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(video_input)
    conc4x0rv = Concatenate(axis=4)([video_input, conv_x40rv])   
    conv_y40rv = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc4x0rv)
    conc4y0rv = Concatenate(axis=4)([video_input, conv_y40rv])
    conv_z40rv = Conv3D(filters=nb_neurons, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc4y0rv)
    conc4z0rv = Concatenate(axis=4)([video_input, conv_z40rv])
    mp_z40rv = MaxPooling3D(pool_size=(32,32,32), padding='same')(conc4z0rv)
    st_y40rv = Conv3D(filters=nb_neurons, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(video_input)
    conc40rstv = Concatenate(axis=4)([video_input, st_y40rv]) 
    mp_st40rv = MaxPooling3D(pool_size=(32,32,32), padding='same')(conc40rstv)
    out_40rv = add([mp_st40rv, mp_z40rv])
    out_c40rv = Conv3D(filters=nb_neurons*32, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_40rv)
    out_c40rvx = BatchNormalization()(out_c40rv)
    
    
    # BLOCK 4: Video 4 RESIDUAL 01
    conv_x41rv = Conv3D(filters=nb_neurons*2, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1vx)
    conc41rxv = Concatenate(axis=4)([out_c1vx, conv_x41rv])
    conv_y41rv = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc41rxv)
    conc41ryv = Concatenate(axis=4)([out_c1vx, conv_y41rv])
    conv_z41rv = Conv3D(filters=nb_neurons*2, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc41ryv)
    conc41rzv = Concatenate(axis=4)([out_c1vx, conv_z41rv])
    mp_z41rv = MaxPooling3D(pool_size=(16,16,16), padding='same')(conc41rzv)
    st_y41rv = Conv3D(filters=nb_neurons*2, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c1vx)
    conc41rstv = Concatenate(axis=4)([out_c1vx, st_y41rv])
    mp_st41rv = MaxPooling3D(pool_size=(16,16,16), padding='same')(conc41rstv)
    out_41rv = add([mp_st41rv, mp_z41rv])
    out_c41rv = Conv3D(filters=nb_neurons*32, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_41rv)
    out_c41rvx = BatchNormalization()(out_c41rv)
    
    # BLOCK 4: Video 4 RESIDUAL 02
    conv_x42rv = Conv3D(filters=nb_neurons*2, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2vx)
    conc42rxv = Concatenate(axis=4)([out_c2vx, conv_x42rv])
    conv_y42rv = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc42rxv)
    conc42ryv = Concatenate(axis=4)([out_c2vx, conv_y42rv])
    conv_z42rv = Conv3D(filters=nb_neurons*2, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc42ryv)
    conc42rzv = Concatenate(axis=4)([out_c2vx, conv_z42rv])
    mp_z42rv = MaxPooling3D(pool_size=(8, 8, 8), padding='same')(conc42rzv)
    st_y42rv = Conv3D(filters=nb_neurons*2, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c2vx)
    conc42rstv = Concatenate(axis=4)([out_c2vx, st_y42rv])
    mp_st42rv = MaxPooling3D(pool_size=(8,8,8), padding='same')(conc42rstv)
    out_42rv = add([mp_st42rv, mp_z42rv])
    out_c42rv = Conv3D(filters=nb_neurons*32, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_42rv)
    out_c42rvx = BatchNormalization()(out_c42rv)
    
    # BLOCK 4: Video 4 RESIDUAL 03
    conv_x43rv = Conv3D(filters=nb_neurons*2, kernel_size=(1, 5, 5), padding='same', activation='relu', kernel_initializer='he_normal')(out_c3vx)
    conc43rxv = Concatenate(axis=4)([out_c3vx, conv_x43rv])
    conv_y43rv = Conv3D(filters=1, kernel_size=(5, 5, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc43rxv)
    conc43ryv = Concatenate(axis=4)([out_c3vx, conv_y43rv])
    conv_z43rv = Conv3D(filters=nb_neurons*2, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc43ryv)
    conc43rzv = Concatenate(axis=4)([out_c3vx, conv_z43rv])
    mp_z43rv = MaxPooling3D(pool_size=(4, 4, 4), padding='same')(conc43rzv)
    st_y43rv = Conv3D(filters=nb_neurons*2, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c3vx)
    conc43rstv = Concatenate(axis=4)([out_c3vx, st_y43rv])
    mp_st43rv = MaxPooling3D(pool_size=(4,4,4), padding='same')(conc43rstv)
    out_43rv = add([mp_st43rv, mp_z43rv])
    out_c43rv = Conv3D(filters=nb_neurons*32, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_43rv)
    out_c43rvx = BatchNormalization()(out_c43rv)
    
      
    # BLOCK 5: Video Main
    conv_x5v = Conv3D(filters=nb_neurons*16, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_c4vx)
    conc5xv = Concatenate(axis=4)([out_c4vx, conv_x5v])
    conv_y5v = Conv3D(filters=1, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(conc5xv)
    conc5yv = Concatenate(axis=4)([out_c4vx, conv_y5v])
    conv_z5v = Conv3D(filters=nb_neurons*16, kernel_size=(1, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(conc5yv)
    conc5zv = Concatenate(axis=4)([out_c4vx, conv_z5v])
    mp_z5v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc5zv)
    st_y5v = Conv3D(filters=nb_neurons*16, kernel_size=(3, 3, 1), padding='same', activation='relu', kernel_initializer='he_normal')(out_c4vx)
    conc5stv = Concatenate(axis=4)([out_c4vx, st_y5v])
    mp_st5v = MaxPooling3D(pool_size=(2,2,2), padding='same')(conc5stv)
    #out_5v = add([mp_st5v, mp_z5v])
    out_5v = add([mp_st5v, mp_z5v, out_c40rvx, out_c41rvx, out_c42rvx, out_c43rvx])
    out_c5v = Conv3D(filters=nb_neurons*32, kernel_size=(3, 3, 3), padding='same', activation='relu', kernel_initializer='he_normal')(out_5v)
    out_c5vx = BatchNormalization()(out_c5v)
    
    
    
    
    #******************************MUSIC 2D Network (GLOCAL Net) **********************************************************#
    audio_shape_phase = (128, 1292, 1)
    nb_neurons = 16
    activation="sigmoid"

    audio_input_mag = Input(shape=audio_shape_phase)
     
    # BLOCK 1: AUDIO Mag
    conv_x1a = Conv2D(filters= nb_neurons, kernel_size=(1, 5), padding='same', activation=activation, kernel_initializer='he_normal')(audio_input_mag)
    conc1xa = Concatenate(axis=3)([audio_input_mag, conv_x1a])
    conv_y1a = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc1xa)
    conc1ya = Concatenate(axis=3)([audio_input_mag, conv_y1a])
    conv_z1a = Conv2D(filters=nb_neurons, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc1ya)
    conc1za = Concatenate(axis=3)([audio_input_mag, conv_z1a])
    mp_z1a = MaxPooling2D(pool_size=(2,2), padding='same')(conc1za)
    st_y1a = Conv2D(filters=nb_neurons, kernel_size=(3, 1), padding='same', activation=activation, kernel_initializer='he_normal')(audio_input_mag)
    conc1sta = Concatenate(axis=3)([audio_input_mag, st_y1a])
    mp_st1a = MaxPooling2D(pool_size=(2,2), padding='same')(conc1sta)
    out_1a = add([mp_st1a, mp_z1a])
    out_c1a = Conv2D(filters=nb_neurons*2, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_1a)
    out_c1ax = BatchNormalization()(out_c1a)
    
    # BLOCK 10: RESIDUAL 
    conv_x10r = Conv2D(filters=nb_neurons, kernel_size=(1, 5), padding='same', activation=activation, kernel_initializer='he_normal')(audio_input_mag)
    conc10xr = Concatenate(axis=3)([audio_input_mag, conv_x10r])
    conv_y10r = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc10xr)
    conc10yr = Concatenate(axis=3)([audio_input_mag, conv_y10r])
    conv_z10r = Conv2D(filters=nb_neurons, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc10yr)
    conc10zr = Concatenate(axis=3)([audio_input_mag, conv_z10r])
    mp_z10r = MaxPooling2D(pool_size=(4,4), padding='same')(conc10zr)
    st_y10r = Conv2D(filters=nb_neurons, kernel_size=(3, 1), padding='same', activation=activation,kernel_initializer='he_normal')(audio_input_mag)
    conc10str = Concatenate(axis=3)([audio_input_mag, st_y10r])
    mp_st10r = MaxPooling2D(pool_size=(4,4), padding='same')(conc10str)
    out_10r = add([mp_st10r, mp_z10r])
    out_c10r = Conv2D(filters=nb_neurons*4, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_10r)
    out_c10rx = BatchNormalization()(out_c10r)

    # BLOCK 2: AUDIO Mag
    conv_x2a = Conv2D(filters=nb_neurons*2, kernel_size=(1, 5), padding='same', activation=activation, kernel_initializer='he_normal')(out_c1ax)
    conc2xa = Concatenate(axis=3)([out_c1ax, conv_x2a])
    conv_y2a = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc2xa)
    conc2ya = Concatenate(axis=3)([out_c1ax, conv_y2a])
    conv_z2a = Conv2D(filters=nb_neurons*2, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc2ya)
    conc2za = Concatenate(axis=3)([out_c1ax, conv_z2a])
    mp_z2a = MaxPooling2D(pool_size=(2,2), padding='same')(conc2za)
    st_y2a = Conv2D(filters=nb_neurons*2, kernel_size=(3, 1), padding='same', activation=activation,kernel_initializer='he_normal')(out_c1ax)
    conc2sta = Concatenate(axis=3)([out_c1ax, st_y2a])
    mp_st2a = MaxPooling2D(pool_size=(2,2), padding='same')(conc2sta)    
    out_2a = add([mp_st2a, mp_z2a, out_c10rx])
    out_c2a = Conv2D(filters=nb_neurons*4, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_2a)
    out_c2ax = BatchNormalization()(out_c2a)
    
    # BLOCK 20: RESIDUAL  MAG
    conv_x20r = Conv2D(filters=nb_neurons, kernel_size=(1, 5), padding='same', activation=activation, kernel_initializer='he_normal')(audio_input_mag)
    conc20xr = Concatenate(axis=3)([audio_input_mag, conv_x20r])
    conv_y20r = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc20xr)
    conc20yr = Concatenate(axis=3)([audio_input_mag, conv_y20r])
    conv_z20r = Conv2D(filters=nb_neurons, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc20yr)
    conc20zr = Concatenate(axis=3)([audio_input_mag, conv_z20r])
    mp_z20r = MaxPooling2D(pool_size=(8,8), padding='same')(conc20zr)
    st_y20r = Conv2D(filters=nb_neurons, kernel_size=(3, 1), padding='same', activation=activation,kernel_initializer='he_normal')(audio_input_mag)
    conc20str = Concatenate(axis=3)([audio_input_mag, st_y20r])
    mp_st20r = MaxPooling2D(pool_size=(8,8), padding='same')(conc20str)
    out_20r = add([mp_st20r, mp_z20r])
    out_c20r = Conv2D(filters=nb_neurons*8, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_20r)
    out_c20rx = BatchNormalization()(out_c20r)
        
    # BLOCK 21: RESIDUAL MAG
    conv_x21r = Conv2D(filters=nb_neurons*2, kernel_size=(1, 5), padding='same', activation=activation, kernel_initializer='he_normal')(out_c1ax)
    conc21xr = Concatenate(axis=3)([out_c1ax, conv_x21r])
    conv_y21r = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc21xr)
    conc21yr = Concatenate(axis=3)([out_c1ax, conv_y21r])
    conv_z21r = Conv2D(filters=nb_neurons*2, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc21yr)
    conc21zr = Concatenate(axis=3)([out_c1ax, conv_z21r])
    mp_z21r = MaxPooling2D(pool_size=(4,4), padding='same')(conc21zr)
    st_y21r = Conv2D(filters=nb_neurons*2, kernel_size=(3, 1), padding='same', activation=activation,kernel_initializer='he_normal')(out_c1ax)
    conc21str = Concatenate(axis=3)([out_c1ax, st_y21r])
    mp_st21r = MaxPooling2D(pool_size=(4,4), padding='same')(conc21str)
    out_21r = add([mp_st21r, mp_z21r])
    out_c21r = Conv2D(filters=nb_neurons*8, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_21r)
    out_c21rx = BatchNormalization()(out_c21r)
  
    # BLOCK 3: AUDIO Mag
    conv_x3a = Conv2D(filters=nb_neurons*4, kernel_size=(1, 5), padding='same', activation=activation, kernel_initializer='he_normal')(out_c2ax)
    conc3xa = Concatenate(axis=3)([out_c2ax, conv_x3a])
    conv_y3a = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc3xa)
    conc3ya = Concatenate(axis=3)([out_c2ax, conv_y3a])
    conv_z3a = Conv2D(filters=nb_neurons*4, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc3ya)
    conc3za = Concatenate(axis=3)([out_c2ax, conv_z3a])
    mp_z3a = MaxPooling2D(pool_size=(2,2), padding='same')(conc3za)
    st_y3a = Conv2D(filters=nb_neurons*4, kernel_size=(3, 1), padding='same', activation=activation, kernel_initializer='he_normal')(out_c2ax)
    conc3sta = Concatenate(axis=3)([out_c2ax, st_y3a])
    mp_st3a = MaxPooling2D(pool_size=(2,2), padding='same')(conc3sta)
    out_3a = add([mp_st3a, mp_z3a, out_c20rx, out_c21rx])
    out_c3a = Conv2D(filters=nb_neurons*8, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_3a)
    out_c3ax = BatchNormalization()(out_c3a)
    
    # BLOCK 30: RESIDUAL MAG
    conv_x30r = Conv2D(filters=nb_neurons, kernel_size=(1, 5), padding='same', activation=activation, kernel_initializer='he_normal')(audio_input_mag)
    conc30xr = Concatenate(axis=3)([audio_input_mag, conv_x30r])
    conv_y30r = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc30xr)
    conc30yr = Concatenate(axis=3)([audio_input_mag, conv_y30r])
    conv_z30r = Conv2D(filters=nb_neurons, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc30yr)
    conc30zr = Concatenate(axis=3)([audio_input_mag, conv_z30r])
    mp_z30r = MaxPooling2D(pool_size=(16,16), padding='same')(conc30zr)
    st_y30r = Conv2D(filters=nb_neurons, kernel_size=(3, 1), padding='same', activation=activation,kernel_initializer='he_normal')(audio_input_mag)
    conc30str = Concatenate(axis=3)([audio_input_mag, st_y30r])
    mp_st30r = MaxPooling2D(pool_size=(16,16), padding='same')(conc30str)
    out_30r = add([mp_st30r, mp_z30r])
    out_c30r = Conv2D(filters=nb_neurons*16, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_30r)
    out_c30rx = BatchNormalization()(out_c30r)
    #print("The shape of Res 30", out_c30rx.shape)
    
    # BLOCK 31: RESIDUAL MAG
    conv_x31r = Conv2D(filters=nb_neurons*2, kernel_size=(1, 5), padding='same', activation=activation, kernel_initializer='he_normal')(out_c1ax)
    conc31xr = Concatenate(axis=3)([out_c1ax, conv_x31r])
    conv_y31r = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc31xr)
    conc31yr = Concatenate(axis=3)([out_c1ax, conv_y31r])
    conv_z31r = Conv2D(filters=nb_neurons*2, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc31yr)
    conc31zr = Concatenate(axis=3)([out_c1ax, conv_z31r])
    mp_z31r = MaxPooling2D(pool_size=(8,8), padding='same')(conc31zr)
    st_y31r = Conv2D(filters=nb_neurons*2, kernel_size=(3, 1), padding='same', activation=activation,kernel_initializer='he_normal')(out_c1ax)
    conc31str = Concatenate(axis=3)([out_c1ax, st_y31r])
    mp_st31r = MaxPooling2D(pool_size=(8,8), padding='same')(conc31str)
    out_31r = add([mp_st31r, mp_z31r])
    out_c31r = Conv2D(filters=nb_neurons*16, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_31r)
    out_c31rx = BatchNormalization()(out_c31r)
    #print("The shape of Res 31", out_c31rx.shape)
    
    # BLOCK 32: RESIDUAL MAG
    conv_x32r = Conv2D(filters=nb_neurons*4, kernel_size=(1, 5), padding='same', activation=activation, kernel_initializer='he_normal')(out_c2ax)
    conc32xr = Concatenate(axis=3)([out_c2ax, conv_x32r])
    conv_y32r = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc32xr)
    conc32yr = Concatenate(axis=3)([out_c2ax, conv_y32r])
    conv_z32r = Conv2D(filters=nb_neurons*4, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc32yr)
    conc32zr = Concatenate(axis=3)([out_c2ax, conv_z32r])
    mp_z32r = MaxPooling2D(pool_size=(4,4), padding='same')(conc32zr)
    st_y32r = Conv2D(filters=nb_neurons*4, kernel_size=(3, 1), padding='same', activation=activation,kernel_initializer='he_normal')(out_c2ax)
    conc32str = Concatenate(axis=3)([out_c2ax, st_y32r])
    mp_st32r = MaxPooling2D(pool_size=(4,4), padding='same')(conc32str)
    out_32r = add([mp_st32r, mp_z32r])
    out_c32r = Conv2D(filters=nb_neurons*16, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_32r)
    out_c32rx = BatchNormalization()(out_c32r)

    # BLOCK 4: AUDIO Mag
    conv_x4a = Conv2D(filters=nb_neurons*8, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_c3ax)
    conc4xa = Concatenate(axis=3)([out_c3ax, conv_x4a])
    conv_y4a = Conv2D(filters=1, kernel_size=(3, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc4xa)
    conc4ya = Concatenate(axis=3)([out_c3ax, conv_y4a])
    conv_z4a = Conv2D(filters=nb_neurons*8, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc4ya)
    conc4za = Concatenate(axis=3)([out_c3ax, conv_z4a])
    mp_z4a = MaxPooling2D(pool_size=(2,2), padding='same')(conc4za)
    st_y4a = Conv2D(filters=nb_neurons*8, kernel_size=(3, 1), padding='same', activation=activation, kernel_initializer='he_normal')(out_c3ax)
    conc4sta = Concatenate(axis=3)([out_c3ax, st_y4a])
    mp_st4a = MaxPooling2D(pool_size=(2,2), padding='same')(conc4sta)
    out_4a = add([mp_st4a, mp_z4a, out_c30rx, out_c31rx, out_c32rx])
    out_c4a = Conv2D(filters=nb_neurons*16, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_4a)
    out_c4ax = BatchNormalization()(out_c4a)
    
    # BLOCK 40: RESIDUAL MAG
    conv_x40r = Conv2D(filters=nb_neurons, kernel_size=(1, 5), padding='same', activation=activation, kernel_initializer='he_normal')(audio_input_mag)
    conc40xr = Concatenate(axis=3)([audio_input_mag, conv_x40r])
    conv_y40r = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc40xr)
    conc40yr = Concatenate(axis=3)([audio_input_mag, conv_y40r])
    conv_z40r = Conv2D(filters=nb_neurons, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc40yr)
    conc40zr = Concatenate(axis=3)([audio_input_mag, conv_z40r])
    mp_z40r = MaxPooling2D(pool_size=(32,32), padding='same')(conc40zr)
    st_y40r = Conv2D(filters=nb_neurons, kernel_size=(3, 1), padding='same', activation=activation,kernel_initializer='he_normal')(audio_input_mag)
    conc40str = Concatenate(axis=3)([audio_input_mag, st_y40r])
    mp_st40r = MaxPooling2D(pool_size=(32,32), padding='same')(conc40str)
    out_40r = add([mp_st40r, mp_z40r])
    out_c40r = Conv2D(filters=nb_neurons*32, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_40r)
    out_c40rx = BatchNormalization()(out_c40r)
    #print("The shape of Res 30", out_c40rx.shape)
    
    # BLOCK 41: RESIDUAL MAG
    conv_x41r = Conv2D(filters=nb_neurons*2, kernel_size=(1, 5), padding='same', activation=activation, kernel_initializer='he_normal')(out_c1ax)
    conc41xr = Concatenate(axis=3)([out_c1ax, conv_x41r])
    conv_y41r = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc41xr)
    conc41yr = Concatenate(axis=3)([out_c1ax, conv_y41r])
    conv_z41r = Conv2D(filters=nb_neurons*2, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc41yr)
    conc41zr = Concatenate(axis=3)([out_c1ax, conv_z41r])
    mp_z41r = MaxPooling2D(pool_size=(16,16), padding='same')(conc41zr)
    st_y41r = Conv2D(filters=nb_neurons*2, kernel_size=(3, 1), padding='same', activation=activation,kernel_initializer='he_normal')(out_c1ax)
    conc41str = Concatenate(axis=3)([out_c1ax, st_y41r])
    mp_st41r = MaxPooling2D(pool_size=(16,16), padding='same')(conc41str)
    out_41r = add([mp_st41r, mp_z41r])
    out_c41r = Conv2D(filters=nb_neurons*32, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_41r)
    out_c41rx = BatchNormalization()(out_c41r)
    #print("The shape of Res 31", out_c41rx.shape)
    
    # BLOCK 42: RESIDUAL MAG
    conv_x42r = Conv2D(filters=nb_neurons*4, kernel_size=(1, 5), padding='same', activation=activation, kernel_initializer='he_normal')(out_c2ax)
    conc42xr = Concatenate(axis=3)([out_c2ax, conv_x42r])
    conv_y42r = Conv2D(filters=1, kernel_size=(5, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc42xr)
    conc42yr = Concatenate(axis=3)([out_c2ax, conv_y42r])
    conv_z42r = Conv2D(filters=nb_neurons*4, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc42yr)
    conc42zr = Concatenate(axis=3)([out_c2ax, conv_z42r])
    mp_z42r = MaxPooling2D(pool_size=(8,8), padding='same')(conc42zr)
    st_y42r = Conv2D(filters=nb_neurons*4, kernel_size=(3, 1), padding='same', activation=activation,kernel_initializer='he_normal')(out_c2ax)
    conc42str = Concatenate(axis=3)([out_c2ax, st_y42r])
    mp_st42r = MaxPooling2D(pool_size=(8,8), padding='same')(conc42str)
    out_42r = add([mp_st42r, mp_z42r])
    out_c42r = Conv2D(filters=nb_neurons*32, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_42r)
    out_c42rx = BatchNormalization()(out_c42r)
    #print("The shape of Res 32", out_c42rx.shape)
    
    # BLOCK 43: RESIDUAL MAG
    conv_x43r = Conv2D(filters=nb_neurons*8, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_c3ax)
    conc43xr = Concatenate(axis=3)([out_c3ax, conv_x43r])
    conv_y43r = Conv2D(filters=1, kernel_size=(3, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc43xr)
    conc43yr = Concatenate(axis=3)([out_c3ax, conv_y43r])
    conv_z43r = Conv2D(filters=nb_neurons*8, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc43yr)
    conc43zr = Concatenate(axis=3)([out_c3ax, conv_z43r])
    mp_z43r = MaxPooling2D(pool_size=(4,4), padding='same')(conc43zr)
    st_y43r = Conv2D(filters=nb_neurons*8, kernel_size=(3, 1), padding='same', activation=activation, kernel_initializer='he_normal')(out_c3ax)
    conc43str = Concatenate(axis=3)([out_c3ax, st_y43r])
    mp_st43r = MaxPooling2D(pool_size=(4,4), padding='same')(conc43str)
    out_43r = add([mp_st43r, mp_z43r])
    out_c43r = Conv2D(filters=nb_neurons*32, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_43r)
    out_c43rx = BatchNormalization()(out_c43r)


    # BLOCK 5: AUDIO Mag
    conv_x5a = Conv2D(filters=nb_neurons*16, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_c4ax)
    conc5xa = Concatenate(axis=3)([out_c4ax, conv_x5a])
    conv_y5a = Conv2D(filters=1, kernel_size=(3, 1), padding='same', activation=activation, kernel_initializer='he_normal')(conc5xa)
    conc5ya = Concatenate(axis=3)([out_c4ax, conv_y5a])
    conv_z5a = Conv2D(filters=nb_neurons*16, kernel_size=(1, 3), padding='same', activation=activation, kernel_initializer='he_normal')(conc5ya)
    conc5za = Concatenate(axis=3)([out_c4ax, conv_z5a])
    mp_z5a = MaxPooling2D(pool_size=(2,2), padding='same')(conc5za)
    st_y5a = Conv2D(filters=nb_neurons*16, kernel_size=(3, 1), padding='same', activation=activation, kernel_initializer='he_normal')(out_c4ax)
    conc5sta = Concatenate(axis=3)([out_c4ax, st_y5a])
    mp_st5a = MaxPooling2D(pool_size=(2,2), padding='same')(conc5sta)
    out_5a = add([mp_st5a, mp_z5a, out_c40rx, out_c41rx, out_c42rx, out_c43rx])
    out_c5a = Conv2D(filters=nb_neurons*32, kernel_size=(3, 3), padding='same', activation=activation, kernel_initializer='he_normal')(out_5a)
    out_c5ax = BatchNormalization()(out_c5a)
    
    
    #********************************************* END MUSIC NETWORK ******************************************************#
    gap_2D_Audio = GlobalAveragePooling2D()(out_c5ax)
    gap_3D_video = GlobalAveragePooling3D()(out_c5vx)
 
    final_out = concatenate([gap_3D_video, gap_2D_Audio])
    final_out = Dense(6, activation='softmax', name='AV_OUT')(final_out)
    
    model = Model(inputs=[video_input, audio_input_mag], outputs=[final_out])
    model.summary()
    
    #model.load_weights('/VideoMMTM_Music2D_model.h5', by_name = True)
    #print("The pre-trained weight loaded.")
    
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.SGD(lr=1e-3),metrics=["accuracy"])  

    return model
    
