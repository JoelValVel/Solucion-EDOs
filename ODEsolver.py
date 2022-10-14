import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
import matplotlib.pyplot as plt
import numpy as np


class ODEsolver(Sequential):
    """Use y_pred for y variable, x for x and dy for dy/dx"""
    def __init__(self, funcion, ic, minval, maxval, **kwargs):
        super().__init__(**kwargs)
        self.loss_tracker = keras.metrics.Mean(name='loss')
        self.funcion = funcion
        self.ic = ic
        self.minval = minval
        self.maxval = maxval

    @property
    def metrics(self):
        return [self.loss_tracker]

    def train_step(self, data):
        batch_size = tf.shape(data)[0]
        x = tf.random.uniform((batch_size,1),minval=self.minval,maxval=self.maxval)

        with tf.GradientTape() as tape:
            #calcula el valor del loss
            with tf.GradientTape() as tape2:
                tape2.watch(x)
                with tf.GradientTape() as tape3:
                    tape3.watch(x)
                    y_pred = self(x,training=True)
                dy = tape3.gradient(y_pred, x)
            dy2 = tape2.gradient(dy,x)
            x_o = tf.zeros([batch_size,1])
            y_o = self(x_o, training=True)
            eq = eval(self.funcion)
            ic = y_o - self.ic # Default initial condition: y(0)=self.ic
            loss = keras.losses.mean_squared_error(0., eq) + keras.losses.mean_squared_error(0., ic)
        # Aplicar gradientes
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        #actualizar metricas
        self.loss_tracker.update_state(loss)
        #regresar diccionario mapeando nombre de las metricas con su valor actual
        return {'loss': self.loss_tracker.result()}


