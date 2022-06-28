import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer
from sklearn.manifold import LocallyLinearEmbedding
from tools_ae import *
from visualisation import *
"""
AutoEncoder IDEA 1 : Each frame is like an image, no temporality, the output is only the positions of KP for one frame, and the creation
of animation is made by chosing the beginning and the ending position and interpolating in the latent space between the two

"""






def AE(latent_size=16):
    ## Définition de l'architecture du modèle
    hidden_size = 256
    hidden_size_1 = 128
    hidden_size_2 = 64
    latent_size = latent_size
    input_layer = tf.keras.layers.Input(shape = (15,2))
    flattened = tf.keras.layers.Flatten()(input_layer)
    hidden = tf.keras.layers.Dense(hidden_size, activation = tf.keras.activations.linear)(flattened)
    hidden = tf.keras.layers.BatchNormalization()(hidden)
    hidden = tf.keras.layers.Dropout(0.2)(hidden)
    hidden_1 = tf.keras.layers.Dense(hidden_size_1, activation = 'relu')(hidden)
    hidden_1 = tf.keras.layers.BatchNormalization()(hidden_1)
    hidden_1 = tf.keras.layers.Dropout(0.2)(hidden_1)
    hidden_2 = tf.keras.layers.Dense(hidden_size_2, activation = 'relu')(hidden_1)
    hidden_2 = tf.keras.layers.BatchNormalization()(hidden_2)
    hidden_2 = tf.keras.layers.Dropout(0.2)(hidden_2)
    latent = tf.keras.layers.Dense(latent_size, activation = 'relu')(hidden_2)
    encoder = tf.keras.Model(inputs = input_layer, outputs = latent, name = 'encoder')
    encoder.summary()

    input_layer_decoder = tf.keras.layers.Input(shape = encoder.output.shape[1:])
    upsampled = tf.keras.layers.Dense(hidden_size, activation = 'relu')(input_layer_decoder)
    upsampled = tf.keras.layers.BatchNormalization()(upsampled)
    upsampled = tf.keras.layers.Dropout(0.2)(upsampled)
    upsampled_1 = tf.keras.layers.Dense(hidden_size_2, activation = 'relu')(upsampled)
    upsampled_1 = tf.keras.layers.BatchNormalization()(upsampled_1)
    upsampled_1 = tf.keras.layers.Dropout(0.2)(upsampled_1)
    upsampled_2 = tf.keras.layers.Dense(hidden_size_1, activation = 'relu')(upsampled_1)
    upsampled_2 = tf.keras.layers.BatchNormalization()(upsampled_2)
    upsampled_2 = tf.keras.layers.Dropout(0.2)(upsampled_2)
    upsampled = tf.keras.layers.Dense(encoder.layers[1].output_shape[-1], activation = tf.keras.activations.linear)(upsampled_2)
    constructed = tf.keras.layers.Reshape(x_train.shape[1:])(upsampled)
    decoder = tf.keras.Model(inputs = input_layer_decoder, outputs = constructed, name= 'decoder')
    decoder.summary()

    autoencoder = tf.keras.Model(inputs = encoder.input, outputs = decoder(encoder.output))
    autoencoder.summary()


    sgd = tf.keras.optimizers.Adam()

    autoencoder.compile(sgd, loss='mse', metrics=['accuracy'])

    return autoencoder,encoder,decoder




if __name__ == '__main__':

    x_train=normalize(load_data(name ="./data/kp_3.npy"))
    visu_skel(x_train,1005)

    autoencoder,encoder,decoder = AE(latent_size = 16)

    autoencoder.fit(x_train,
         x_train,
         batch_size=128,
         epochs=10
         )

    autoencoder.save("./models/model_15")
    encoder.save("./models/model_15_encoder")
    decoder.save("./models/model_15_decoder")

    # autoencoder = tf.keras.models.load_model('./model_1')
    # encoder = tf.keras.models.load_model('./model_1_encoder')
    # decoder = tf.keras.models.load_model('./model_1_decoder')