from keras.layers import Dense,Dropout,Flatten,BatchNormalization,Conv2D,Conv2DTranspose,Concatenate,LeakyReLU,Activation
from keras.optimizers import Adam
from keras.models import Input,Model
from keras.initializers import RandomNormal

#discriminator takes in real input,sampled from dataset and generated input
#produces a 30x30x1 tensor
#PatchGAN architecture
image_shape = (256,256,3)


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

discriminator = define_discriminator(image_shape)
discriminator.summary()

def define_encoder(input_tensor,n_filters,batchnorm = True):
    init = RandomNormal(stddev = 0.2)

    x = Conv2D(n_filters,(4,4),strides = (2,2),padding = 'same',kernel_initializer = init)(input_tensor)
    if batchnorm:
        x = BatchNormalization()(x,training = True)
    x = LeakyReLU(0.2)(x)
    return x

def define_decoder(input_tensor,n_filters,skip_tensor,dropout = True):
    init = RandomNormal(stddev = 0.2)

    x = Conv2DTranspose(n_filters,(4,4),strides = (2,2),padding = 'same',kernel_initializer = init)(input_tensor)
    x = BatchNormalization()(x,training = True)
    if dropout:
        x = Dropout(0.4)(x,training = True)
    x = Activation('relu')(x)
    return x

def define_generator(image_shape):

    init = RandomNormal(stddev = 0.2)
    real_input = Input(shape = image_shape)

    e1 = define_encoder(real_input,64,batchnorm = False)
    e2 = define_encoder(e1,128)
    e3 = define_encoder(e2,256)
    e4 = define_encoder(e3,512)
    e5 = define_encoder(e4,512)
    e6 = define_encoder(e5,512)
    e7 = define_encoder(e6,512)
    # e8 = define_encoder(e7,512)
    b = Conv2D(512,(2,2),strides = 1,padding = 'valid',kernel_initializer = init)(e7)
    # b = BatchNormalization()(b)
    b = Activation('relu')(b)

    d1 = define_decoder(b,512,e7)
    d2 = define_decoder(d1,512,e6)
    d3 = define_decoder(d2,512,e5)
    d4 = define_decoder(d3,512,e4,dropout = False)
    d5 = define_decoder(d4,256,e3,dropout = False)
    d6 = define_decoder(d5,128,e2,dropout = False)
    d7 = define_decoder(d6,64,e1,dropout = False)

    out = Conv2DTranspose(3,(4,4),strides = (2,2),padding = 'same',kernel_initializer = init)(d7)
    out = Activation('tanh')(out)

    x = Model(real_input,out)

    return x

g = define_generator(image_shape)
g.summary()
