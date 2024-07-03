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
import scipy
from scipy import linalg
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tqdm import tqdm


from IPython import display

BUFFER_SIZE = 320 #define el número de muestras a partir de las cuales se elige aleatoriamente durante el entrenamiento.
BATCH_SIZE = 16 #es el número de muestras de entrenamiento que se propagan a través de la red a la vez._
LEARNING_RATE = 2e-4
NOISE_DIM = 100
EPOCHS = 3703 
num_examples_to_generate = 4
seed = tf.random.normal([num_examples_to_generate, NOISE_DIM])

def make_generator():
    model = tf.keras.Sequential()
    model.add(layers.Dense(16*16*128, use_bias=False, input_shape=(NOISE_DIM,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 128)))
    assert model.output_shape == (None, 16, 16, 128)

    model.add(layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(1, (3, 3), strides=(1, 1), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 128, 128, 1)

    return model

def make_discriminator():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[128, 128, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1, activation='sigmoid'))

    return model


def total_variation_loss(image):
    x_deltas = image[:, 1:, :, :] - image[:, :-1, :, :]
    y_deltas = image[:, :, 1:, :] - image[:, :, :-1, :]
    return tf.reduce_mean(tf.abs(x_deltas)) + tf.reduce_mean(tf.abs(y_deltas))

def success_rate(r, FAR, FIPS, t):
    return 1 - (1 - r * FAR) ** (FIPS * t)

generator = make_generator()
noise = tf.random.normal([1, NOISE_DIM])
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

lambda_tv = 0.001

def generator_loss(fake_output, generated_images):
    gan_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
    tv_loss = total_variation_loss(generated_images)
    total_loss = gan_loss + lambda_tv * tv_loss
    return total_loss


generator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
discriminator_optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)

checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)



@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      noise = tf.random.normal(shape=tf.shape(generated_images), mean=0.0, stddev=0.1)
      generated_images_with_noise = generated_images + noise

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images_with_noise, training=True)
      gen_loss = generator_loss(fake_output, generated_images_with_noise)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

def train(images, epochs):
    generator_losses = []
    discriminator_losses = []
    for epoch in range(epochs):
        start = time.time()

        for image_batch in images:
            gen_loss, disc_loss = train_step(image_batch)
            generator_losses.append(gen_loss)
            discriminator_losses.append(disc_loss)

        display.clear_output(wait=True)
        generate_and_save_images(generator, epoch + 1, seed, saveLast=True)

        # Guardar el modelo cada 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('\nTime for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

    display.clear_output(wait=True)
    generate_and_save_images(generator, epochs, seed, saveLast=True)
    return generator_losses, discriminator_losses

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


input_img_path = "./DB"

files_names = os.listdir(input_img_path)
images = []

file_name = files_names[0]  # 

for file_name in files_names:
    image = cv2.imread(input_img_path + "/" + file_name, 0)
    if image is not None:  
        image_normalized = (image - 127.5) / 127.5 
        image_redim = cv2.resize(image_normalized, (128, 128))
        images.append(image_redim)
    else:
        print("Error al cargar la imagen:", file_name)


images_np = np.array(images)

train_dataset = tf.data.Dataset.from_tensor_slices(images_np).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

generator_losses, discriminator_losses = train(train_dataset, EPOCHS)

plt.figure(figsize=(10, 5))
plt.plot(generator_losses, label='Generator Loss', alpha=0.7)
plt.plot(discriminator_losses, label='Discriminator Loss', alpha=0.7)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Generator and Discriminator Losses over Time')
plt.legend()
plt.grid(True)
plt.show()


# 1. Obtener imágenes reales del conjunto de datos
real_images = images_np  # Utilizamos las imágenes cargadas en 'images_np'

# 2. Generar imágenes con el generador
noise = tf.random.normal([num_examples_to_generate, NOISE_DIM])
generated_images = generator(noise, training=False)

# Función para calcular las características de las imágenes utilizando InceptionV3
def calculate_activation_statistics(image_redim, model):
    # Asegurar que las imágenes tengan la forma adecuada para la red discriminatoria
    image_redim = tf.reshape(image_redim, [-1, 128, 128, 1])
    
    # Duplicar el canal de la imagen 3 veces para crear una imagen en color falsa
    image_redim = tf.repeat(image_redim, 3, axis=-1)

    # Redimensionar las imágenes a (299, 299) para el modelo InceptionV3
    images_resized = tf.image.resize(image_redim, (299, 299))
    
    # Preprocesar las imágenes de acuerdo con las expectativas de InceptionV3
    images_preprocessed = preprocess_input(images_resized)
    
    # Calcular las características utilizando el modelo InceptionV3
    features = model.predict(images_preprocessed)

    # Calcular la media y la matriz de covarianza de las características
    mu = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    
    return mu, sigma



# Función para calcular la distancia de Frechet entre dos distribuciones de características
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2):
    epsilon = 1e-6
    mu_diff = mu1 - mu2
    cov_mean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(cov_mean).all():
        cov_mean = np.zeros_like(cov_mean)
    distance = np.sum(mu_diff**2) + np.trace(sigma1 + sigma2 - 2*cov_mean)
    return np.real(distance)

# Función para calcular el FID entre imágenes reales y generadas
def calculate_fid(real_images, generated_images, model):
    real_features = calculate_activation_statistics(real_images, model)
    generated_features = calculate_activation_statistics(generated_images, model)
    fid = calculate_frechet_distance(real_features[0], real_features[1], generated_features[0], generated_features[1])
    return fid

# Cargar el modelo InceptionV3 preentrenado
inception_model = InceptionV3(include_top=False, pooling='avg', input_shape=(299, 299, 3))

# Calcular el FID entre las imágenes reales y generadas
fid_score = calculate_fid(image_redim, generated_images, inception_model)
print("FID Score:", fid_score)

