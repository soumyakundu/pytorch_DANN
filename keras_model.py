import numpy as np

def getModelGivenModelOptionsAndWeightInits(init_weights=None,seed=1234):
    np.random.seed(seed)
    import keras;
    from keras.models import Sequential
    from keras.layers.core import Dropout, Reshape, Dense, Activation, Flatten
    from keras.layers.convolutional import Conv2D, MaxPooling2D
    from keras.optimizers import Adadelta, SGD, RMSprop;
    import keras.losses;
    from keras.constraints import maxnorm;
    from keras.layers.normalization import BatchNormalization
    from keras.regularizers import l1, l2
    from keras import backend as K
    from keras.layers import Input
    from keras.models import Model

    K.set_image_data_format('channels_last')
    print(K.image_data_format())

    inputs = Input(shape=(1,1000,4))

    if (init_weights!=None):
        #load the weight initializations
        data=np.load(init_weights);
        x = Conv2D(filters=300,kernel_size=(1,19),weights=[data['0.Conv/weights:0'],np.zeros(300,)],padding="same", name='conv1')(inputs)
        x = BatchNormalization(axis=-1,weights=[data['2.BatchNorm/gamma:0'],data['1.BatchNorm/beta:0'],np.zeros(300,),np.zeros(300,)], name='bn1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)
        x = Conv2D(filters=200,kernel_size=(1,11),weights=[data['3.Conv_1/weights:0'],np.zeros(200,)],padding="same", name='conv2')(x)
        x = BatchNormalization(axis=-1,weights=[data['5.BatchNorm_1/gamma:0'],data['4.BatchNorm_1/beta:0'],np.zeros(200,),np.zeros(200,)], name='bn2')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1,4))(x)
        x = Conv2D(filters=200,kernel_size=(1,7),weights=[data['6.Conv_2/weights:0'],np.zeros(200,)],padding="same", name='conv3')(x)
        x = BatchNormalization(axis=-1,weights=[data['8.BatchNorm_2/gamma:0'],data['7.BatchNorm_2/beta:0'],np.zeros(200,),np.zeros(200,)], name='bn3')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1,4))(x)

        x = Flatten()(x)
        x = Dense(1000,weights=[data['9.fc0/fully_connected/weights:0'],np.zeros(1000,)], name='fc1')(x)
        x = BatchNormalization(axis=1,weights=[data['11.fc0/BatchNorm/gamma:0'],data['10.fc0/BatchNorm/beta:0'],np.zeros(1000,),np.zeros(1000,)], name='bn4')(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

        x = Dense(1000,weights=[data['12.fc1/fully_connected/weights:0'],np.zeros(1000,)], name='fc2')(x)
        x = BatchNormalization(axis=1,weights=[data['14.fc1/BatchNorm/gamma:0'],data['13.fc1/BatchNorm/beta:0'],np.zeros(1000,),np.zeros(1000,)], name='bn5')(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

        x = Dense(1, name='fc3')(x)
        outputs = Activation("sigmoid")(x)

    else:
        x = Conv2D(filters=300,kernel_size=(1,19),input_shape=(1,1000,4), padding='same', name='conv1')(inputs)
        x = BatchNormalization(axis=-1, name='bn1')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1,3))(x)

        x = Conv2D(filters=200,kernel_size=(1,11), padding='same', name='conv2')(x)
        x = BatchNormalization(axis=-1, name='bn2')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1,4))(x)

        x = Conv2D(filters=200,kernel_size=(1,7), padding='same', name='conv3')(x)
        x = BatchNormalization(axis=-1, name='bn3')(x)
        x = Activation('relu')(x)
        x = MaxPooling2D(pool_size=(1,4))(x)

        x = Flatten()(x)
        x = Dense(1000, name='fc1')(x)
        x = BatchNormalization(axis=-1, name='bn4')(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

        x = Dense(1000, name='fc2')(x)
        x = BatchNormalization(axis=-1, name='bn5')(x)
        x = Activation('relu')(x)
        x = Dropout(0.3)(x)

        x = Dense(1, name='fc3')(x)
        outputs = Activation("sigmoid")(x)

    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    print("compiling!")
    loss="binary_crossentropy"
    model = Model(inputs = inputs, outputs = outputs)
    model.compile(optimizer=adam,loss=loss)
    return model
