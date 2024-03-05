import tensorflow as tf
from tensorflow.keras.losses import BinaryCrossentropy
tf.__version__
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os 
from PIL import Image
from tensorflow.keras import layers
import time


from IPython import display

BUFFER_SIZE = 60000
BATCH_SIZE = 256

def make_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(7*7*512, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 512)))
    assert model.output_shape == (None, 7, 7, 512)

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 7, 7, 256)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 14, 14, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 28, 28, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 28, 28, 1)

    return model

def make_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same',
                                     input_shape=[28, 28, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

def success_rate(r, FAR, FIPS, t):
    return 1 - (1 - r * FAR) ** (FIPS * t)

generator = make_generator()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)

plt.imshow(generated_image[0, :, :, 0], cmap='gray')
cross_entropy = BinaryCrossentropy(from_logits=True)
discriminator = make_discriminator()
decision = discriminator(generated_image)
print (decision)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

EPOCHS = 400
noise_dim = 100
num_examples_to_generate = 16
seed = tf.random.normal([num_examples_to_generate, noise_dim])

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def train(images, epochs):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in images:
            train_step(image_batch)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed, saveLast=True)

        # Guardar el modelo cada 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('\nTime for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed, saveLast=True)

def generate_and_save_images(model, epoch, test_input, saveLast=False):
    predictions = model(test_input, training=False)

    if saveLast and epoch < EPOCHS:
        return

    fig = plt.figure(figsize=(10, 10))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(predictions[i, :, :, 0] * 0.5 + 0.5, cmap='gray')  
        plt.axis('off')

    # Save the generated images to a file
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.show()



input_img_path = "C:/Users/alici/Documents/Uni/TFG/reading_images"
files_names = os.listdir(input_img_path)
images = []

file_name = files_names[0]  # Escoge la primera imagen de la lista

image = cv2.imread(input_img_path + "/" + file_name, 0)
print(input_img_path + "/" + file_name)
image_normalized = (image - 127.5) / 127.5 
image_redim = cv2.resize(image_normalized, (28, 28))
# for file_name in files_names:
#     image = cv2.imread(input_img_path + "/" + file_name, 0)
#     print(input_img_path + "/" + file_name)
#     image_normalized = (image - 127.5) / 127.5 
#     image_redim = cv2.resize(image_normalized, (28, 28))
#     images.append(image_redim)

# images_np = np.array(images)
images = [image_redim]
# Entrenar el modelo
#train_dataset = tf.data.Dataset.from_tensor_slices(images_np).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
train_dataset = tf.data.Dataset.from_tensor_slices(np.array(images)).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

train(train_dataset, EPOCHS)

# Cargar y mostrar la última imagen generada
last_generated_image = Image.open('last_generated_image.png')
plt.imshow(last_generated_image)
plt.axis('off')
plt.show()


r = 1  # Número de huellas dactilares inscritas en el dispositivo víctima
FAR = 0.99  # Tasa de falsa aceptación del sistema objetivo
FIPS = 10  # Número de imágenes de huellas dactilares enviadas por segundo
t = 5  # Número de intentos de fuerza bruta

# Calcular la tasa de éxito
rate = success_rate(r, FAR, FIPS, t)
print("Tasa de éxito del ataque de fuerza bruta:", rate)
#print("Tasa de éxito del ataque después de {} segundos: {:.2%}".format(t, rate))