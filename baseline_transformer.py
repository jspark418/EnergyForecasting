# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import sys
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error
from model import Transformer # 모델 추가

ENERGY = sys.argv[1]

print('baseline_{}'.format(ENERGY))
for num in range(0,5) :
    print('{}_running'.format(num))
    
    # data load
    def preprocessing(data) :
        feature = data.copy()[['time','elec','water','gas','hotwater','hot']]
        feature.time = pd.to_datetime(feature.time)
        feature.set_index('time',inplace=True)
        y = data[[ENERGY]]
        return feature, y

    train = pd.read_csv('train_summer.csv')
    valid = pd.read_csv('valid_summer.csv')
    test = pd.read_csv('test_summer.csv')

    X_train, y_train = preprocessing(train)
    X_valid, y_valid = preprocessing(valid)
    X_test, y_test = preprocessing(test)

    scaler = MinMaxScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train))
    X_valid = pd.DataFrame(scaler.transform(X_valid))
    X_test = pd.DataFrame(scaler.transform(X_test))


    # make time window
    def timeseries_data(dataset, target, start_index, end_index, window_size, target_size) :
        data = []
        labels = []

        y_start_index = start_index + window_size # 0+24
        y_end_index = end_index - target_size  # train_index(10291) - 24 = 10267

        for i in range(y_start_index, y_end_index) :
            data.append(dataset.iloc[i-window_size:i,:].values)
            labels.append(target.iloc[i:i+target_size,:].values)
        data = np.array(data)
        labels = np.array(labels)
        labels = labels.reshape(-1,target_size)  
        return data, labels

    window = 72
    X_train, y_train = timeseries_data(X_train,y_train,0,len(X_train),window,24)
    X_valid, y_valid = timeseries_data(X_valid,y_valid,0,len(X_valid),window,24)
    X_test, y_test = timeseries_data(X_test,y_test,0,len(X_test),window,24)

    batch_size = 32
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(X_train.shape[0],seed=42).batch(batch_size, drop_remainder=True).prefetch(1)
    valid_ds = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(batch_size, drop_remainder=True).prefetch(1)
    test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(batch_size, drop_remainder=True).prefetch(1)
     
    # model architecture
    class AttentionModel(keras.Model):

        def __init__(self,name="attentionmodel"):
            super(AttentionModel, self).__init__(name=name)
            self.encoder1 = Transformer()
            self.encoder2 = Transformer()
            self.encoder3 = Transformer()
            self.encoder4 = Transformer()
            self.encoder5 = Transformer()
            self.flatten = keras.layers.Flatten()
            self.hidden1 = keras.layers.Dense(512, kernel_initializer='he_normal',activation = 'relu')
            self.hidden2 = keras.layers.Dense(256, kernel_initializer= 'he_normal',activation = 'relu')
            self.output_ = keras.layers.Dense(24, kernel_initializer= 'he_normal')

        def call(self, input1, input2, input3, input4, input5, target, training):
            out1 = self.encoder1(tf.expand_dims(input1,2),tf.expand_dims(target,2), training)
            out2 = self.encoder2(tf.expand_dims(input2,2),tf.expand_dims(target,2), training)
            out3 = self.encoder3(tf.expand_dims(input3,2),tf.expand_dims(target,2), training)
            out4 = self.encoder4(tf.expand_dims(input4,2),tf.expand_dims(target,2), training)
            out5 = self.encoder5(tf.expand_dims(input5,2),tf.expand_dims(target,2), training)
            out1 = tf.expand_dims(out1,1)
            out2 = tf.expand_dims(out2,1)
            out3 = tf.expand_dims(out3,1)
            out4 = tf.expand_dims(out4,1)
            out5 = tf.expand_dims(out5,1)
            concat_energy = tf.concat([out1,out2,out3,out4,out5],axis=1)
            flatten = self.flatten(concat_energy)
            hidden1 = self.hidden1(flatten)
            hidden2 = self.hidden2(hidden1)
            output = self.output_(hidden2)

            return output
    
    # save best model
    class EarlyStopping:
        """Early stops the training if validation loss doesn't improve after a given patience."""
        def __init__(self, patience=30, verbose=False, delta=0):
            """
            Args:
                patience (int): How long to wait after last time validation loss improved.
                                Default: 7
                verbose (bool): If True, prints a message for each validation loss improvement. 
                                Default: False
                delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                                Default: 0
            """
            self.patience = patience
            self.verbose = verbose
            self.counter = 0
            self.best_score = None
            self.early_stop = False
            self.val_loss_min = np.Inf
            self.delta = delta

        def __call__(self, val_loss, model):

            score = -val_loss

            if self.best_score is None:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
            elif score < self.best_score + self.delta:
                self.counter += 1
                # print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                # if self.counter >= self.patience:
                #     self.early_stop = True
            else:
                self.best_score = score
                self.save_checkpoint(val_loss, model)
                self.counter = 0
            
            return self.early_stop

        def save_checkpoint(self, val_loss, model):
            '''Saves model when validation loss decrease.'''
            # if self.verbose:
            #     print(f'loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            print(f'loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).')
            model.save_weights('./checkpoints/my_checkpoint')
            self.val_loss_min = val_loss

    # learning rate scheduler
    class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
        def __init__(self, d_model, warmup_steps=4000):
            super(CustomSchedule, self).__init__()
            self.d_model = d_model
            self.d_model = tf.cast(self.d_model, tf.float32)
            self.warmup_steps = warmup_steps

        def __call__(self, step):
            arg1 = tf.math.rsqrt(step)
            arg2 = step * (self.warmup_steps ** -1.5)
            return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

    # training hyperparamters
    model = AttentionModel()
    n_epochs = 500
    learning_rate = CustomSchedule(d_model=32)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    loss_fn_1 = keras.losses.MeanSquaredError()
    loss_fn_test = keras.losses.MeanSquaredError()
    metric = keras.metrics.MeanAbsoluteError()
    early = EarlyStopping()


    # training loop
    print('training start')
    out1_df = []
    concat_df = []
    result_df = pd.DataFrame({'MAPE' : [], 'MAE' : [], 'RMSE' : []})
    loss_df = []
    loss_test = []
    val_loss = []
    for epoch in range(n_epochs) :
        loss_batch = 0
        for batch,(features, label) in train_ds.enumerate() :
            with tf.GradientTape() as tape :
                y_pred = model(features[:,:,0],features[:,:,1],features[:,:,2],
                                                     features[:,:,3], features[:,:,4], label, training=True)
                loss = loss_fn_1(label, y_pred)
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            loss_batch += loss
        loss_df.append(loss_batch)
        # valid set
        val_metric = 0
        sk_metric = 0
        for batch,(features, label) in valid_ds.enumerate() :
            y_pred = model(features[:,:,0],features[:,:,1],features[:,:,2],features[:,:,3],features[:,:,4], 
                       label, training=False)
            error_mae = mean_absolute_error(label, y_pred)
            sk_metric+=error_mae
        print('sk',sk_metric,'epoch',epoch)
        val_loss.append(sk_metric)
        if early(sk_metric, model):
            print(early(sk_metric, model))
            break
    
    # test model performance
    model.load_weights('./checkpoints/my_checkpoint')
    predict_df = []
    real_df = []
    loss_ = 0
    for batch,(features, label) in test_ds.enumerate() :
        y_pred = model(features[:,:,0],features[:,:,1],features[:,:,2],features[:,:,3],features[:,:,4], 
                       label, training=False)
        loss_+=loss_fn_test(label, y_pred)
        predict_df.append(y_pred)
        real_df.append(label)
    loss_test.append(loss_)

    real_new = [a.numpy() for b in real_df for a in b]
    predict_new = [a.numpy().round(1) for b in predict_df for a in b] 


    error_mape = mean_absolute_percentage_error(real_new, predict_new)
    error_mae = mean_absolute_error(real_new, predict_new)
    error_rmse = mean_squared_error(real_new, predict_new)**(0.5)
    result = pd.DataFrame({'MAPE' : [error_mape], 'MAE' : [error_mae], 'RMSE' : [error_rmse]})
    result_df = pd.concat([result_df,result],axis=0)
    print(result_df)
    
    result_df.to_pickle('./base_{}/result_{}.csv'.format(ENERGY, num))
    loss_df = [i.numpy() for i in loss_df]
    pd.DataFrame(loss_df).to_pickle('./base_{}/loss_{}.csv'.format(ENERGY, num))
    loss_test = [i.numpy() for i in loss_test]
    pd.DataFrame(loss_test).to_pickle('./base_{}/loss_test_{}.csv'.format(ENERGY, num))
    pd.DataFrame(val_loss).to_pickle('./base_{}/loss_val_{}.csv'.format(ENERGY, num))
