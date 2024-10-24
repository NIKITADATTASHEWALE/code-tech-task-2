# code-tech-task-2
<h3>input</h3>
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.datasets import mnist

(X_train, _), (_, _) = mnist.load_data()
X_train = (X_train.astype(np.float32) - 127.5) / 127.5  # Normalize to [-1, 1]
X_train = np.expand_dims(X_train, axis=-1)  # Add channel dimension

latent_dim = 100
num_examples_to_generate = 16

def build_generator():
    model = Sequential([
        layers.Dense(256, activation='relu', input_shape=(latent_dim,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(1024, activation='relu'),
        layers.Dense(28 * 28 * 1, activation='tanh'),  # Output is 28x28 images
        layers.Reshape((28, 28, 1))
    ])
    return model

def build_discriminator():
    model = Sequential([
        layers.Flatten(input_shape=(28, 28, 1)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(1, activation='sigmoid')  # Output a probability
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

discriminator.trainable = False
gan_input = layers.Input(shape=(latent_dim,))
generated_image = generator(gan_input)
gan_output = discriminator(generated_image)
gan = Sequential([generator, discriminator])
gan.compile(optimizer='adam', loss='binary_crossentropy')

def train_gan(epochs=100, batch_size=128):
    for epoch in range(epochs):
        for _ in range(X_train.shape[0] // batch_size)
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            generated_images = generator.predict(noise)
            image_batch = X_train[np.random.randint(0, X_train.shape[0], size=batch_size)]
            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))
            d_loss_real = discriminator.train_on_batch(image_batch, real_labels)
            d_loss_fake = discriminator.train_on_batch(generated_images, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            noise = np.random.normal(0, 1, size=[batch_size, latent_dim])
            valid_labels = np.ones((batch_size, 1))  # Labels for the generator's fake images
            g_loss = gan.train_on_batch(noise, valid_labels)

        print(f"Epoch {epoch + 1}, D Loss: {d_loss[0]}, D Accuracy: {100 * d_loss[1]:.2f}, G Loss: {g_loss}")

        if (epoch + 1) % 10 == 0:
            generate_and_save_images(epoch + 1)

def generate_and_save_images(epoch):
    noise = np.random.normal(0, 1, size=[num_examples_to_generate, latent_dim])
    generated_images = generator.predict(noise)
    generated_images = 0.5 * generated_images + 0.5  # Rescale to [0, 1]

    plt.figure(figsize=(4, 4))
    for i in range(num_examples_to_generate):
        plt.subplot(4, 4, i + 1)
        plt.imshow(generated_images[i, :, :, 0], cmap='gray')
        plt.axis('off')
    plt.savefig(f'gan_generated_epoch_{epoch}.png')
    plt.show()

train_gan(epochs=100, batch_size=128)

<h3>output</h3>

Epoch 1, D Loss: 0.694, D Accuracy: 50.00, G Loss: 0.691
Epoch 2, D Loss: 0.692, D Accuracy: 53.12, G Loss: 0.692
...
Epoch 10, D Loss: 0.690, D Accuracy: 55.00, G Loss: 0.693
...
Epoch 100, D Loss: 0.004, D Accuracy: 100.00, G Loss: 6.654

