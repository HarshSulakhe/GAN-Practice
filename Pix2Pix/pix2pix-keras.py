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

# discriminator.summary()

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

# generator.summary()

def define_gan(discriminator,generator,image_shape):
    discriminator.trainable = False
    input = Input(shape = image_shape)
    gen_out = generator(input)

    dis_out = discriminator([input,gen_out])

    model = Model(input,[dis_out,gen_out])

    opt = Adam(lr = 0.0002,beta_1 = 0.5)
    model.compile(optimizer = opt, loss = ['binary_crossentropy','mae'],loss_weights = [1,100])
    return model


# gan.summary()


def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

def load_real_samples(filename):
	# load compressed arrays
	data = load(filename)
	# unpack arrays
	X1, X2 = data['arr_0'], data['arr_1']
	# scale from [0,255] to [-1,1]
	X1 = (X1 - 127.5) / 127.5
	X2 = (X2 - 127.5) / 127.5
	return [X1, X2]


# generate samples and save as a plot and save the model
def summarize_performance(step, g_model, dataset, n_samples=3):
	# select a sample of input images
	[X_realA, X_realB], _ = generate_real_samples(dataset, n_samples, 1)
	# generate a batch of fake samples
	X_fakeB, _ = generate_fake_samples(g_model, X_realA, 1)
	# scale all pixels from [-1,1] to [0,1]
	X_realA = (X_realA + 1) / 2.0
	X_realB = (X_realB + 1) / 2.0
	X_fakeB = (X_fakeB + 1) / 2.0
	# plot real source images
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realA[i])
	# plot generated target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples + i)
		pyplot.axis('off')
		pyplot.imshow(X_fakeB[i])
	# plot real target image
	for i in range(n_samples):
		pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
		pyplot.axis('off')
		pyplot.imshow(X_realB[i])
	# save plot to file
	filename1 = 'plot_%06d.png' % (step+1)
	pyplot.savefig(filename1)
	pyplot.close()
	# save the generator model
	filename2 = 'model_%06d.h5' % (step+1)
	g_model.save(filename2)
	print('>Saved: %s and %s' % (filename1, filename2))

#training algo
# generate a real sample pair A,B from a dataset.
# Feed it to discriminator with target as 1s
# generate a fake B using A and feed to discriminator with target as 0s
# Feed realA,[1s,realB] to gan model it will compare with 1s with disc(realA) with bce loss, and compare realB with gen(A)
# with L1 Loss

def train_gan(discriminator,generator,gan,dataset,n_epochs = 100,n_batch = 1,n_patch = 30):
    trainA, trainB = dataset
    batches = int(len(trainA)/n_batch)
    for i in range(n_epochs*batches):
        X_realA,X_realB,y_real = generate_real_samples(dataset,n_batch,n_patch)
        X_fakeB,y_fake = generate_fake_samples(X_realA,generator,n_patch)
        d1_loss = discriminator.train_on_batch([X_realA,X_realB],y_real)
        d2_loss = discriminator.train_on_batch([X_realA,X_fakeB],y_fake)
        g_loss,_,_ = gan.train_on_batch(X_realA,[y_real,X_realB])
        print('>%d, d1[%.3f] d2[%.3f] g[%.3f]' % (i+1, d_loss1, d_loss2, g_loss))
		# summarize model performance
		if (i+1) % (bat_per_epo * 10) == 0:
			summarize_performance(i, g_model, dataset)


image_shape = (256,256,3)
dataset = load_real_samples('maps_256.npz')
generator = define_generator(image_shape)
discriminator = define_discriminator(image_shape)
gan = define_gan(discriminator,generator,image_shape)
train_gan(d_model, g_model, gan_model, dataset)
