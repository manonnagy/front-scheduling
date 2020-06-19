import numpy as np
import random
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.losses import MeanSquaredError, MSE
from collections import deque

class Agent():
    def __init__(self, input_size, output_size):
        tf.keras.backend.clear_session()
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.9999995
        self.epsilon_min = 0.01
        self.learning_rate = 0.002
        self.rho = 0.9
        self.input_size = input_size
        self.output_size = output_size
        self.sequential_replay = deque([], maxlen = 2000)
        
        input_layer = keras.Input(shape=(self.input_size,), name="obs")
        hidden = Dense(32, activation="elu", kernel_initializer = "he_uniform")(input_layer)
        hidden2 = Dense(32, activation="elu")(hidden)
        output_layer = Dense(self.output_size, activation="linear")(hidden2)
        self.model = keras.Model(inputs=input_layer, outputs=output_layer)
        
        opt = RMSprop(learning_rate=self.learning_rate, rho=self.rho)
        #opt = Adam(self.learning_rate)
        self.optimizer = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")
        #self.loss_func = MeanSquaredError(reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE)
        self.loss_func = MSE
        
    def update_sequential_replay(self, state, action, reward, done, next_state):
        self.sequential_replay.append((state, action, reward, done, next_state))
        return 0
    
    def train(self, batch_size = 32, epochs = 1, use_loss_scale = False, use_grad_clip = False):
        if len(self.sequential_replay) < batch_size:
            return
        for _ in range(epochs):
            states, q_targets, actions = self.get_prepared_batch(batch_size)
            self.compute_grad(states, q_targets, actions, use_loss_scale, use_grad_clip)
        return 0
    
    def get_prepared_batch(self, batch_size):        
        batch_indices = np.random.randint(0, high = len(self.sequential_replay), size = batch_size)
        batch_buffer = [self.sequential_replay[ind] for ind in batch_indices]                           
        states, actions, rewards, dones, next_states = [np.array([obs[ind] for obs in batch_buffer]) for ind in range(5)]
        
        states = tf.constant(states, dtype="float32")
        rewards = tf.reshape(tf.constant(rewards.T, dtype="float32"),(batch_size,1))
        next_states = tf.constant(next_states, dtype="float32")
        
        q_targets = tf.math.add(rewards,self.gamma * tf.reduce_max(self.model(next_states), axis=1, keepdims=True))
        return states, q_targets, actions
        
    @tf.function
    def compute_grad(self, states, q_targets, actions, use_loss_scale, grad_clip):
        gradients = None
        mask = tf.one_hot(actions, self.output_size)
        with tf.GradientTape() as tape: 
            q_value = self.model(states)
            q_value_choose = tf.reduce_sum(mask*q_value, axis = 1, keepdims = True)
            loss = self.loss_func(q_targets, q_value_choose)
            scaled_loss = self.optimizer.get_scaled_loss(loss)
        if use_loss_scale:
            scaled_grad = tape.gradient(scaled_loss, self.model.trainable_variables)
            gradients = self.optimizer.get_unscaled_gradients(scaled_grad)
        else:
            gradients = tape.gradient(loss, self.model.trainable_variables)
        if grad_clip:
            gradients = [tf.clip_by_norm(g, grad_clip) for g in gradients]
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return 0
        
    def get_best_action(self, state, rand=True):
        self.epsilon *= self.epsilon_decay if rand else 1

        if rand and np.random.rand() <= self.epsilon and self.epsilon > self.epsilon_min:
            return random.randrange(self.output_size)
        
        state = tf.constant(state)
        act_values = self.model(state)

        action = tf.math.argmax(act_values[0]).numpy()
        return action

    def model_saving(self, name = "model"):
        model_json = self.model.to_json() # serialize model to JSON
        with open(name+".json", "w") as json_file:
            json_file.write(model_json)
        
        self.model.save_weights(name+".h5") # serialize weights to HDF5
        print("Saved model to disk")
        return 0
    
    
    def model_loading(self, name = "model"):
        json_file = open(name+".json", 'r') # load json and create model
        loaded_model_json = json_file.read()
        json_file.close()
        self.model = tf.keras.models.model_from_json(loaded_model_json)
        
        self.model.load_weights(name+".h5") # load weights into new model
        print("Loaded model from disk")
        return 0
