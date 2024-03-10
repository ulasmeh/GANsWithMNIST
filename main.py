import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import time
import os
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Reshape, ReLU,Convolution2DTranspose, Flatten, LeakyReLU,Conv2D, Concatenate, Input
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers.legacy import Adam  #M1,M2 mimarisinde daha hızlı çalıştığı için
from tensorflow.keras.utils import to_categorical

# load train dataset
(x_train, y_train),(x_test,y_test) = tf.keras.datasets.mnist.load_data()

# concat train and test feature. do not consider labels
X_train_concat = np.concatenate((x_train,x_test))


train_size = X_train_concat.shape[0]
train_set = X_train_concat.reshape(train_size, 28, 28, 1).astype("float32")

train_set = (train_set - 127.5)/127.5

y_train_concat = np.concatenate((y_train, y_test))
y_train_categorical = tf.one_hot(y_train_concat, 10)
label_set = tf.reshape(y_train_categorical,(train_size, 10, 1))

batch_size = 140
train_data_set = tf.data.Dataset.from_tensor_slices((train_set,label_set)).shuffle(train_size).batch(batch_size)
# train_data_set represents the preprocessed data ready for training, consisting of batches of size 128 and shuffled.

num_classes = 10

def conditional_generator_model():
    initializer = tf.keras.initializers.HeNormal(seed = 42)
    noise = Input(shape=(100, ))
    label = Input(shape=(10,))
    x = Concatenate()([noise, label])
    x = Dense(7*7*256, kernel_initializer = initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = Reshape((7,7,256))(x)

    x = Convolution2DTranspose(128, (5, 5), strides = (1, 1), padding = "same", kernel_initializer = initializer )(x) # When padding is set to "same", the output size becomes equal to the input size multiplied by the stride.
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    x = Convolution2DTranspose(64, (5, 5), strides = (2, 2), padding = "same", kernel_initializer = initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Convolution2DTranspose(32, (5, 5), strides = (2, 2), padding = "same", kernel_initializer = initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    out = Convolution2DTranspose(1, (5, 5), strides = (1, 1), padding = "same", activation="tanh", kernel_initializer = initializer)(x) # The activation function is tanh because the real image data has been scaled between [-1, 1].
    
    model = Model([noise,label], out)
    return model

def conditional_discriminator_model():
    initializer = tf.keras.initializers.GlorotNormal(seed = 42)

    label = Input(shape=(10,))
    image_input = Input(shape=(28, 28, 1))

    x = Conv2D(32, kernel_size = (5,5), strides = (1, 1), padding = "same", kernel_initializer = initializer )(image_input) #The channel size is halved because the strides parameter is 2.
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(64, kernel_size = (5,5), strides = (2, 2), padding ="same", kernel_initializer = initializer )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)

    x = Conv2D(128, kernel_size = (5,5), strides = (2, 2), padding ="same", kernel_initializer = initializer )(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)

    x_concat_layer = Concatenate()([x, label])


    output_layer = Dense(1, activation = "sigmoid" )(x_concat_layer)
    model = Model([image_input, label], output_layer)
    return model

con_discriminator = conditional_discriminator_model()
con_discriminator.summary()

# Loss
bce = BinaryCrossentropy()
def discriminator_loss(real_output, fake_output):
    real_loss = bce(tf.ones_like(real_output), real_output ) # Discriminator'un gerçek deme olasılığı ile gerçekte doğru olan örnekler arasında edilen loss
    fake_loss = bce(tf.zeros_like(fake_output), fake_output) # Discriminator'un fake deme olasılığı ile gerçekte fake olan örnekler arasında edilen loss
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    gen_loss = bce(tf.ones_like(fake_output), fake_output) #Generator fake image'ları doğru gibi gösterme ölçüsünce başarılıdır. Bu sebeple gerçekte fake olan örneklerin gerçek olma durumu arasındaki loss ölçülür
    return gen_loss

# Optimizer
generator_optimizer = Adam(learning_rate = 0.002, beta_1 = 0.5)
discriminator_optimizer = Adam(learning_rate = 0.002, beta_1 = 0.5)

epochs = 50
noise_dim = 100
num_examples_to_generate = 16

seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images, labels):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = con_generator([noise, labels], training = True)

        real_output = con_discriminator([images, labels], training = True)
        fake_output = con_discriminator([generated_images, labels], training = True)

        disc_loss = discriminator_loss(real_output, fake_output)
        gen_loss = generator_loss(fake_output)
    
    gradients_of_generator = gen_tape.gradient(gen_loss, con_generator.trainable_variables )
    gradietns_of_discriminator = disc_tape.gradient(disc_loss, con_discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, con_generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradietns_of_discriminator, con_discriminator.trainable_variables))

    return (gen_loss, disc_loss, tf.reduce_mean(real_output), tf.reduce_mean(fake_output))

checkpoint_dir = "./Conditional_training_checkpoints"

checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer, 
                                 discriminator_optimizer = discriminator_optimizer, 
                                 con_discriminator = con_discriminator, 
                                 con_generator = con_generator
                                )
def generate_and_plot_images(model, epoch, test_input):
    predictions = model(test_input, training = False)
    
    fig = plt.figure(figsize = (8, 4))
    
    for i in range(predictions.shape[0]):
        plt.subplot(4, 4 , i + 1)
        pred = (predictions[i, :,:,0]+ 1 )*127.5
        pred = np.array(pred)
        plt.imshow(pred.astype(np.uint8), cmap = "gray")
        plt.axis("off")
        
    plt.savefig("ConGANs/image_at_epoch_{:04d}.png".format(epoch))
    plt.show()

epochs = 50
noise_dim = 100
num_examples_to_generate = 16
noise = tf.random.normal([num_examples_to_generate, noise_dim])

seed = [noise, label_set[:num_examples_to_generate]]

print("denenen boyutlar")
print(noise.shape)
print(label_set[:num_examples_to_generate].shape)

def train(dataset, epochs):
    gen_loss_list = []
    disc_loss_list = []
    
    
    real_score_list = []
    fake_score_list = []
    
    for epoch in tqdm(range(epochs)):
        start = time.time()
        num_batches = len(dataset)
        
        
        print(f"Training started with epoch {epoch + 1} with {num_batches} batches...")
        
        total_gen_loss = 0
        total_disc_loss = 0
        
        for (X,y) in dataset:
            generator_loss, discriminator_loss, real_score, fake_score = train_step(X,y)
            total_gen_loss += generator_loss
            total_disc_loss += discriminator_loss
            #print(epoch,total_disc_loss)
        print(total_gen_loss)
        print(total_disc_loss)
        print(num_batches)


        mean_gen_loss = total_gen_loss/num_batches
        mean_disc_loss = total_disc_loss/num_batches

        print(mean_gen_loss)
        print(mean_disc_loss)
        
        print("Losses after epoch %5d: generator %.3f, discriminator %.3f, real_score %.2f%%, fake_score %2f%% " %
             (epoch +1, generator_loss, discriminator_loss, real_score*100,fake_score*100))
        
        
        generate_and_plot_images(con_generator, epoch +1, seed)
        
        gen_loss_list.append(mean_gen_loss)
        disc_loss_list.append(mean_disc_loss)
        real_score_list.append(real_score)
        fake_score_list.append(fake_score)
        print(gen_loss_list)
        
        if (epoch +1)%10 ==0:
            checkpoint.save(file_prefix=checkpoint_prefix)
            
        print("Time for epoch {} is {} sec".format(epoch +1, time.time()-start))
        
    return gen_loss_list, disc_loss_list, real_score_list, fake_score_list

gen_loss_epochs, disc_loss_epochs, real_score_list, fake_score_list = train(train_data_set, epochs = 50)