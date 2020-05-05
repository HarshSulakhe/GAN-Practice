from keras.layers import Dense,Dropout,Flatten,BatchNormalization,Conv2D,Conv2DTranspose,Concatenate,LeakyReLU,Activation
from keras.optimizers import Adam
from keras.models import Input,Model
from keras.initializers import RandomNormal

#discriminator takes in real input,sampled from dataset and generated input
#produces a 30x30x1 tensor
#PatchGAN architecture
def define_discriminator(image_shape):
    #initialize weights with std = 0.2
    init = RandomNormal(stddev = 0.2)
    real_input = Input(shape = image_shape)
    gen_input = Input(shape = image_shape)

    merged = Concatenate()([real_input,gen_input])

    x = Conv2D(64,(4,4),strides = (2,2),padding = 'same',kernel_initializer = init)(merged)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(128,(4,4),strides = (2,2),padding = 'same',kernel_initializer = init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(256,(4,4),strides = (2,2),padding = 'same',kernel_initializer = init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(512,(2,2),strides = (1,1),padding = 'valid',kernel_initializer = init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(1,(2,2),strides = (1,1),padding = 'valid',kernel_initializer = init)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    out = Activation('sigmoid')(x)

    model = Model([real_input,gen_input],out)
    opt = Adam(lr = 0.0002,beta_1 = 0.5)
    model.compile(optimizer = opt,loss = 'binary_crossentropy',loss_weights = [0.5])
    return model

image_shape = (256,256,3)

discriminator = define_discriminator(image_shape)

discriminator.summary()
