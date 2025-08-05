import numpy as np
import random
import tensorflow 
import tensorflow as tf
random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)
from tensorflow import keras
from tensorflow.keras import Input, Model,layers, models
from keras.models import load_model
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.callbacks import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
import warnings 
warnings.filterwarnings(action='ignore')

class MemoryModule(tf.keras.layers.Layer):
    def __init__(self, memory_size, feature_dim, shrink_thres=0.002, **kwargs):
        super().__init__(**kwargs)
        self.memory_size = memory_size
        self.feature_dim = feature_dim
        self.shrink_thres = shrink_thres
        self.memory = self.add_weight(shape=(memory_size, feature_dim),initializer='uniform',trainable=True,name='memory')
    def call(self, z):
        z_norm = l2_normalize(z)
        m_norm = l2_normalize(self.memory)
        sim = tf.matmul(z_norm, tf.transpose(m_norm))
        w = tf.nn.softmax(sim, axis=1)
        w_hat = hard_shrinkage(w, lamb=self.shrink_thres)
        z_hat = tf.matmul(w_hat, self.memory)
        return z_hat, w_hat
    def get_config(self):
        config = super().get_config()
        config.update({'memory_size': self.memory_size,'feature_dim': self.feature_dim,'shrink_thres': self.shrink_thres})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
    
def l2_normalize(x, axis=-1, epsilon=1e-10):
    x = tf.convert_to_tensor(x)
    if len(x.shape) == 1:
        x = tf.expand_dims(x, 0)  # (latent_dim,) → (1, latent_dim)
    return x / (K.sqrt(K.sum(K.square(x), axis=axis, keepdims=True)) + epsilon)

def hard_shrinkage(w, lamb=0.002, epsilon=1e-12):
    w = tf.nn.relu(w - lamb) * w / (tf.abs(w - lamb) + epsilon)
    return w / (K.sum(w, axis=1, keepdims=True) + epsilon)

def entropy_loss(w_hat):
    return -K.sum(w_hat * K.log(w_hat + 1e-12), axis=-1)

def memae_loss(x,x_recon):
    x_hat = x_recon[0]
    w_hat = x_recon[1]
    recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(x - x_hat), axis=1))
    entropy = tf.reduce_mean(entropy_loss(w_hat))
    return recon_loss + 0.0002 * entropy    
      
def build_mem_encoder(input_dim,hidden_1,hidden_2,latent_dim,dropout_ratio):
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    inputs = Input(shape=(input_dim,))
    x = Dense(hidden_1, activation='relu')(inputs)
    x = Dense(hidden_2, activation='relu')(x)
    x = Dropout(dropout_ratio)(x)
    z = Dense(latent_dim)(x)
    mem_encoder = Model(inputs,z,name='MemEncoder')
    return mem_encoder

def build_mem_decoder(latent_dim, hidden_2, hidden_1, input_dim, dropout_ratio):
    random.seed(1234)
    np.random.seed(1234)
    tf.random.set_seed(1234)
    z_inputs = Input(shape=(latent_dim,))
    x_hat = Dense(hidden_2, activation='relu')(z_inputs)
    x_hat = Dense(hidden_1, activation='relu')(x_hat)
    x_hat = Dropout(dropout_ratio)(x_hat)
    x_hat = Dense(input_dim, activation='linear')(x_hat)
    mem_decoder = Model(z_inputs,x_hat,name='MemDecoder')
    return mem_decoder

def memae_val_loss(X_val, model, batch_size=64):
    val_losses = []
    for i in range(0, len(X_val), batch_size):
        x_batch = tf.convert_to_tensor(X_val[i:i+batch_size], dtype=tf.float32)
        x_hat, _ = model(x_batch)
        recon_loss = tf.reduce_mean(tf.keras.losses.mae(x_batch, x_hat)).numpy()
        val_losses.append(recon_loss)
    return np.mean(val_losses)

def memae_anomaly_scores(x, model):
    x_hat, _ = model.predict(x, batch_size=64)
    recon_error = np.mean(np.abs(x - x_hat), axis=1)
    return recon_error

def memae_anomaly_scores_extended(x, model):
    x_hat, w_hat = model.predict(x, batch_size=64)
    recon_error = np.mean(np.abs(x - x_hat), axis=1)
    entropy = -np.sum(w_hat * np.log(w_hat + 1e-12), axis=1)
    anomaly_score = recon_error + 0.0002 * entropy
    return anomaly_score