import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np


class ODEsolver(Sequential):
    """Use y_pred for y variable, x for x and dy for dy/dx"""
    def __init__(self, funcion,ic11,ic0,ic1, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.funcion = funcion
        self.ic11 = ic11
        self.ic0 = ic0
        self.ic1 = ic1

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        x = tf.random.uniform((batch_size,1),minval=-1,maxval=1)

        with tf.GradientTape() as tape:
            #calcula el valor del loss
            y_pred = self(x,training=True)
            x_o = tf.zeros([batch_size,1])
            y_o = self(x_o, training=True)
            ic = y_o - self.ic0
            y_11 = self(-tf.ones([batch_size,1]),training = True)
            ic2 = y_11-self.ic11
            y_1 = self(tf.ones([batch_size,1]),training=True)
            ic3 = y_1-self.ic1
            loss = keras.losses.mean_squared_error(eval(self.funcion), y_pred) + keras.losses.mean_squared_error(0., ic) + keras.losses.mean_squared_error(0,ic2) + keras.losses.mean_squared_error(0,ic3)

        # Aplicar gradientes
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #actualizar metricas
        self.loss_tracker.update_state(loss)
        #regresar diccionario mapeando nombre de las metricas con su valor actual
        return {'loss': self.loss_tracker.result()}


