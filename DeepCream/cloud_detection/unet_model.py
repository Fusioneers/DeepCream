from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, concatenate, Conv2DTranspose, Dropout


def unet_model(img_width=256, img_height=192, img_channels=3):
    """

    Returns: A U-Net Model with the following structure (depending on the parameters)

    __________________________________________________________________________________________________
     Layer (type)                   Output Shape         Param #     Connected to
    ==================================================================================================
     input_1 (InputLayer)           [(None, 256, 192, 3  0           []
                                    )]

     conv2d (Conv2D)                (None, 256, 192, 16  448         ['input_1[0][0]']
                                    )

     dropout (Dropout)              (None, 256, 192, 16  0           ['conv2d[0][0]']
                                    )

     conv2d_1 (Conv2D)              (None, 256, 192, 16  2320        ['dropout[0][0]']
                                    )

     max_pooling2d (MaxPooling2D)   (None, 128, 96, 16)  0           ['conv2d_1[0][0]']

     conv2d_2 (Conv2D)              (None, 128, 96, 32)  4640        ['max_pooling2d[0][0]']

     dropout_1 (Dropout)            (None, 128, 96, 32)  0           ['conv2d_2[0][0]']

     conv2d_3 (Conv2D)              (None, 128, 96, 32)  9248        ['dropout_1[0][0]']

     max_pooling2d_1 (MaxPooling2D)  (None, 64, 48, 32)  0           ['conv2d_3[0][0]']

     conv2d_4 (Conv2D)              (None, 64, 48, 64)   18496       ['max_pooling2d_1[0][0]']

     dropout_2 (Dropout)            (None, 64, 48, 64)   0           ['conv2d_4[0][0]']

     conv2d_5 (Conv2D)              (None, 64, 48, 64)   36928       ['dropout_2[0][0]']

     max_pooling2d_2 (MaxPooling2D)  (None, 32, 24, 64)  0           ['conv2d_5[0][0]']

     conv2d_6 (Conv2D)              (None, 32, 24, 128)  73856       ['max_pooling2d_2[0][0]']

     dropout_3 (Dropout)            (None, 32, 24, 128)  0           ['conv2d_6[0][0]']

     conv2d_7 (Conv2D)              (None, 32, 24, 128)  147584      ['dropout_3[0][0]']

     max_pooling2d_3 (MaxPooling2D)  (None, 16, 12, 128)  0          ['conv2d_7[0][0]']

     conv2d_8 (Conv2D)              (None, 16, 12, 256)  295168      ['max_pooling2d_3[0][0]']

     dropout_4 (Dropout)            (None, 16, 12, 256)  0           ['conv2d_8[0][0]']

     conv2d_9 (Conv2D)              (None, 16, 12, 256)  590080      ['dropout_4[0][0]']

     conv2d_transpose (Conv2DTransp  (None, 32, 24, 128)  131200     ['conv2d_9[0][0]']
     ose)

     concatenate (Concatenate)      (None, 32, 24, 256)  0           ['conv2d_transpose[0][0]',
                                                                      'conv2d_7[0][0]']

     conv2d_10 (Conv2D)             (None, 32, 24, 128)  295040      ['concatenate[0][0]']

     dropout_5 (Dropout)            (None, 32, 24, 128)  0           ['conv2d_10[0][0]']

     conv2d_11 (Conv2D)             (None, 32, 24, 128)  147584      ['dropout_5[0][0]']

     conv2d_transpose_1 (Conv2DTran  (None, 64, 48, 64)  32832       ['conv2d_11[0][0]']
     spose)

     concatenate_1 (Concatenate)    (None, 64, 48, 128)  0           ['conv2d_transpose_1[0][0]',
                                                                      'conv2d_5[0][0]']

     conv2d_12 (Conv2D)             (None, 64, 48, 64)   73792       ['concatenate_1[0][0]']

     dropout_6 (Dropout)            (None, 64, 48, 64)   0           ['conv2d_12[0][0]']

     conv2d_13 (Conv2D)             (None, 64, 48, 64)   36928       ['dropout_6[0][0]']

     conv2d_transpose_2 (Conv2DTran  (None, 128, 96, 32)  8224       ['conv2d_13[0][0]']
     spose)

     concatenate_2 (Concatenate)    (None, 128, 96, 64)  0           ['conv2d_transpose_2[0][0]',
                                                                      'conv2d_3[0][0]']

     conv2d_14 (Conv2D)             (None, 128, 96, 32)  18464       ['concatenate_2[0][0]']

     dropout_7 (Dropout)            (None, 128, 96, 32)  0           ['conv2d_14[0][0]']

     conv2d_15 (Conv2D)             (None, 128, 96, 32)  9248        ['dropout_7[0][0]']

     conv2d_transpose_3 (Conv2DTran  (None, 256, 192, 16  2064       ['conv2d_15[0][0]']
     spose)                         )

     concatenate_3 (Concatenate)    (None, 256, 192, 32  0           ['conv2d_transpose_3[0][0]',
                                    )                                 'conv2d_1[0][0]']

     conv2d_16 (Conv2D)             (None, 256, 192, 16  4624        ['concatenate_3[0][0]']
                                    )

     dropout_8 (Dropout)            (None, 256, 192, 16  0           ['conv2d_16[0][0]']
                                    )

     conv2d_17 (Conv2D)             (None, 256, 192, 16  2320        ['dropout_8[0][0]']
                                    )

     conv2d_18 (Conv2D)             (None, 256, 192, 1)  17          ['conv2d_17[0][0]']

    ==================================================================================================
    Total params: 1,941,105
    Trainable params: 1,941,105
    Non-trainable params: 0
    __________________________________________________________________________________________________


    """

    inputs = Input((img_width, img_height, img_channels))
    s = inputs

    # Contraction path
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(s)
    c1 = Dropout(0.1)(c1)
    c1 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c1)
    p1 = MaxPooling2D((2, 2))(c1)

    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p1)
    c2 = Dropout(0.1)(c2)
    c2 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p2)
    c3 = Dropout(0.2)(c3)
    c3 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c3)
    p3 = MaxPooling2D((2, 2))(c3)

    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p3)
    c4 = Dropout(0.2)(c4)
    c4 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c4)
    p4 = MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(p4)
    c5 = Dropout(0.3)(c5)
    c5 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c5)

    # Expansive path
    u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = concatenate([u6, c4])
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c6)

    u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = concatenate([u7, c3])
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c7)

    u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = concatenate([u8, c2])
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c8)

    u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = concatenate([u9, c1], axis=3)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(u9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(c9)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

    model = Model(inputs=inputs, outputs=[outputs])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model