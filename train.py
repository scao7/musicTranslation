from tensorflow.keras.datasets import mnist

from autoencoder import VAE


LEARNING_RATE = 0.0005
BATCH_SIZE = 32
EPOCHS = 100


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype("float32") / 255
    x_train = x_train.reshape(x_train.shape + (1,))
    x_test = x_test.astype("float32") / 255
    x_test = x_test.reshape(x_test.shape + (1,))

    return x_train, y_train, x_test, y_test


def train(x_train, learning_rate, batch_size, epochs):
    autoencoder = VAE(
        input_shape= (28,28,1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=100
    )
    autoencoder.summary()
    autoencoder.compile(learning_rate)
    autoencoder.train(x_train, batch_size, epochs)
    return autoencoder


if __name__ == "__main__":
    # x_train, _, _, _ = load_mnist()
    import os
    import librosa 
    import numpy as np
    from tools import signal_plot,log_spectrumgram_plot
    import cv2
    jazz = "archive/Data/genres_original/jazz"
    x_train =  []
    for filename in os.listdir(jazz):
        # print(filename)
        signal,sr = signal, sr = librosa.load(os.path.join(jazz , filename),sr=22050 )
        # spectrum = log_spectrumgram_plot(signal,2048,512)
        x_train.append([signal])
    # x_train = np.array(x_train)
    # x_train = x_train.astype('float32')/255.
    # x_train = x_train.reshape(len(x_train),np.prod(x_train.shape[1:]))
    
    
    autoencoder = train(x_train, LEARNING_RATE, BATCH_SIZE, EPOCHS)
    autoencoder.save("model")
