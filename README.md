# PRODIGY_GA_04

Steps of the project: 
Here’s a concise version of the Image-to-Image Translation with cGAN project, broken down into clear steps with code:
### 1. **Setup Environment**```python
!pip install tensorflow matplotlib```
### 2. Import Libraries
from tensorflow.keras import layersimport numpy as np
import matplotlib.pyplot as pltfrom tensorflow.keras.datasets import mnist
### 3. **Load and Preprocess Data**```python
(x_train, _), (_, _) = mnist.load_data()x_train = (x_train.astype(np.float32) - 127.5) / 127.5
x_train = np.expand_dims(x_train, axis=3)```
### 4. Define Generator Model
    model = tf.keras.Sequential([        layers.Dense(128, input_dim=100),
        layers.LeakyReLU(alpha=0.2),        layers.Dense(256),
        layers.LeakyReLU(alpha=0.2),        layers.Dense(512),
        layers.LeakyReLU(alpha=0.2),        layers.Dense(28*28*1, activation='tanh'),
        layers.Reshape((28, 28, 1))    ])
    return model
### 5. Define Discriminator Model
    model = tf.keras.Sequential([        layers.Conv2D(64, kernel_size=3, strides=2, padding="same", input_shape=(28, 28, 1)),
        layers.LeakyReLU(alpha=0.2),        layers.Dropout(0.3),
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),        layers.LeakyReLU(alpha=0.2),
        layers.Dropout(0.3),        layers.Flatten(),
        layers.Dense(1, activation='sigmoid')    ])
    return model
### 6. Compile Models
discriminator.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5), metrics=['accuracy'])
generator = build_generator()z = layers.Input(shape=(100,))
img = generator(z)discriminator.trainable = False
valid = discriminator(img)combined = tf.keras.Model(z, valid)
combined.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0002, 0.5))
### 7. Train the cGAN
    real = np.ones((batch_size, 1))    fake = np.zeros((batch_size, 1))
    for epoch in range(epochs):
        # Train Discriminator        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs = x_train[idx]        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict(noise)        d_loss_real = discriminator.train_on_batch(imgs, real)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator        g_loss = combined.train_on_batch(noise, real)
        # Save images at intervals
        if epoch % save_interval == 0:            print(f"{epoch} [D loss: {d_loss[0]}, acc.: {100*d_loss[1]}%] [G loss: {g_loss}]")
            save_imgs(epoch)
def save_imgs(epoch):    noise = np.random.normal(0, 1, (25, 100))
    gen_imgs = generator.predict(noise)
    gen_imgs = 0.5 * gen_imgs + 0.5    fig, axs = plt.subplots(5, 5)
    for i in range(5):        for j in range(5):
            axs[i, j].imshow(gen_imgs[i * 5 + j, :, :, 0], cmap='gray')            axs[i, j].axis('off')
    plt.show()
train(epochs=10000, batch_size=32, save_interval=1000)
This streamlined code covers the essential steps for implementing and training an Image-to-Image Translation model using cGAN.
