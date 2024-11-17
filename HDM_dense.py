import os
import pickle
import numpy as np
import tensorflow.python.keras.layers
 
from config import *
from keras import layers
import tensorflow as tf
import keras
import keras.backend as K
from keras import Model
from keras.layers import Input
from keras.layers import TimeDistributed
from keras.layers import Dense
from keras.layers import LSTM
from tensorflow.keras.callbacks import Callback
from keras.layers import BatchNormalization,LayerNormalization
from keras.layers import Dropout
from keras.layers import concatenate
from keras.callbacks import ModelCheckpoint
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.callbacks import TensorBoard
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
 
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
def create_training_data(corpus_path=CORPUS_PATH, seg_length=SEGMENT_LENGTH, val_ratio=VAL_RATIO):
    # Load corpus
    with open(corpus_path, "rb") as filepath:
        corpus = pickle.load(filepath)
 
    # Inputs and targets for the training set
    input_melody_left = []
    input_melody_right = []
    input_beat_left = []
    input_beat_right = []
    input_fermata_left = []
    input_fermata_right = []
    input_chord_root_left = []
    input_chord_root_right = []
    input_chord_duration_left = []
    input_chord_duration_right = []
    input_chord_intervals_left = []
    input_chord_intervals_right = []
    input_alto = []
    input_tenor = []
    input_bass = []
    output_alto = []
    output_tenor = []
    output_bass = []
 
    # Inputs and targets for the validation set
    val_input_melody_left = []
    val_input_melody_right = []
    val_input_beat_left = []
    val_input_beat_right = []
    val_input_fermata_left = []
    val_input_fermata_right = []
    val_input_chord_root_left = []
    val_input_chord_root_right = []
    val_input_chord_duration_left = []
    val_input_chord_duration_right = []
    val_input_chord_intervals_left = []
    val_input_chord_intervals_right = []
    val_input_chord_left = []
    
    val_input_alto = []
    val_input_tenor = []
    val_input_bass = []
    val_output_alto = []
    val_output_tenor = []
    val_output_bass = []
 
    # Load txt data
    soprano_melody = corpus[0][0]
    soprano_beat = corpus[0][1]
    soprano_fermata = corpus[0][2]
    soprano_chord_root = corpus[0][3]
    soprano_chord_intervals = corpus[0][4]
    soprano_chord_duration = corpus[0][5]
 
    alto_melody = corpus[1]
    tenor_melody = corpus[2]
    bass_melody = corpus[3]
    filenames = corpus[4]
 
    beat_data = soprano_beat
    fermata_data = soprano_fermata
    chord_data1 = soprano_chord_root
    chord_data2 = soprano_chord_intervals
    chord_data3 = soprano_chord_duration
 
    cnt = 0
    np.random.seed(0)
 
    # Process each melody sequence in the corpus
 
    for song_idx in range(len(soprano_melody)):
 
        # Randomly assigned to the training or validation set with the probability
        if np.random.rand() > val_ratio:
            train_or_val = 'train'
 
        else:
            train_or_val = 'val'
            print(filenames[song_idx])
 
        melody_left =   soprano_melody[song_idx] 
        melody_right =   soprano_melody[song_idx][::-1]
        beat_left =   beat_data[song_idx]  
        beat_right =   beat_data[song_idx][::-1]
        fermata_left =   fermata_data[song_idx]  
        fermata_right = fermata_data[song_idx][::-1]
        chord_root_left =   chord_data1[song_idx]  
        chord_root_right =   chord_data1[song_idx][::-1]
        chord_intervals_left =chord_data2[song_idx] 
        chord_intervals_right = chord_data2[song_idx][::-1]
        chord_duration_left =   chord_data3[song_idx]  
        chord_duration_right = chord_data3[song_idx][::-1]
        target_alto =   alto_melody[song_idx]  
 
        target_tenor =   tenor_melody[song_idx]  
        target_bass =   bass_melody[song_idx]  
        
 
        if train_or_val == 'train':
            input_melody_left.append(melody_left)
            input_melody_right.append(melody_right)
            input_beat_left.append(beat_left)
            input_beat_right.append(beat_right)
            input_fermata_left.append(fermata_left)
            input_fermata_right.append(fermata_right)
            input_chord_root_left.append(chord_root_left)
            input_chord_root_right.append(chord_root_right)
            input_chord_intervals_left.append(chord_intervals_left)
            input_chord_intervals_right.append(chord_intervals_right)
            input_chord_duration_left.append(chord_duration_left)
            input_chord_duration_right.append(chord_duration_right)
 
            output_alto.append(target_alto)
            output_tenor.append(target_tenor)
            output_bass.append(target_bass)
 
        else:
            val_input_melody_left.append(melody_left)
            val_input_melody_right.append(melody_right)
            val_input_beat_left.append(beat_left)
            val_input_beat_right.append(beat_right)
            val_input_fermata_left.append(fermata_left)
            val_input_fermata_right.append(fermata_right)
 
            val_input_chord_root_left.append(chord_root_left)
            val_input_chord_root_right.append(chord_root_right)
            val_input_chord_intervals_left.append(chord_intervals_left)
            val_input_chord_intervals_right.append(chord_intervals_right)
            val_input_chord_duration_left.append(chord_duration_left)
            val_input_chord_duration_right.append(chord_duration_right)
 
            val_output_alto.append(target_alto)
            val_output_tenor.append(target_tenor)
            val_output_bass.append(target_bass)
 
        cnt += 1
    print("Successfully read %d samples" % (cnt))
    input_beat_left = np.array(input_beat_left)
    print(input_beat_left.shape)
    input_chord_intervals_left = np.array(input_chord_intervals_left)
    input_chord_intervals_right = np.array(input_chord_intervals_right)
    print(input_chord_intervals_left.shape)
    # One-hot vectorization
    input_melody_left = to_categorical(input_melody_left, num_classes=130)
    input_melody_right = to_categorical(input_melody_right, num_classes=130)
    input_beat_left = to_categorical(input_beat_left, num_classes=4)
    input_beat_right = to_categorical(input_beat_right, num_classes=4)
    input_fermata_left = to_categorical(input_fermata_left, num_classes=2)
    input_fermata_right = to_categorical(input_fermata_right, num_classes=2)
    input_chord_root_left = to_categorical(input_chord_root_left, num_classes=128)
    input_chord_root_right = to_categorical(input_chord_root_right, num_classes=128)
    input_chord_duration_left = to_categorical(input_chord_duration_left, num_classes=20)
    input_chord_duration_right = to_categorical(input_chord_duration_right, num_classes=20)
    output_alto = np.array(to_categorical(output_alto, num_classes=130)).astype(np.float32)
    output_tenor = np.array(to_categorical(output_tenor, num_classes=130)).astype(np.float32)
    output_bass = np.array(to_categorical(output_bass, num_classes=130)).astype(np.float32)
    '''
    output_alto = to_categorical(output_alto, num_classes=130)
    output_tenor = to_categorical(output_tenor, num_classes=130)
    output_bass = to_categorical(output_bass, num_classes=130)
    '''
    # concat beat, fermata and chord
    input_condition_left = np.concatenate((input_beat_left, input_fermata_left), axis=-1)
    input_condition_right = np.concatenate((input_beat_right, input_fermata_right), axis=-1)
    input_chord_left = np.concatenate((input_chord_duration_left,input_chord_intervals_left,input_chord_root_left),axis = -1)
    input_chord_right = np.concatenate((input_chord_duration_right, input_chord_intervals_right, input_chord_root_right),axis=-1)
    '''input_condition_left = np.concatenate((input_beat_left, input_fermata_left), axis=-1)
    input_condition_right = np.concatenate((input_beat_right, input_fermata_right), axis=-1)'''
    if len(val_input_melody_left) != 0:
        val_input_melody_left = to_categorical(val_input_melody_left, num_classes=130)
        val_input_melody_right = to_categorical(val_input_melody_right, num_classes=130)
        val_input_beat_left = to_categorical(val_input_beat_left, num_classes=4)
        val_input_beat_right = to_categorical(val_input_beat_right, num_classes=4)
        val_input_fermata_left = to_categorical(val_input_fermata_left, num_classes=2)
        val_input_fermata_right = to_categorical(val_input_fermata_right, num_classes=2)
        '''
        val_output_alto = to_categorical(val_output_alto, num_classes=130)
        val_output_tenor = to_categorical(val_output_tenor, num_classes=130)
        val_output_bass = to_categorical(val_output_bass, num_classes=130)
        '''
        val_output_alto = np.array(to_categorical(val_output_alto, num_classes=130)).astype(np.float32)
        val_output_tenor = np.array(to_categorical(val_output_tenor, num_classes=130)).astype(np.float32)
        val_output_bass = np.array(to_categorical(val_output_bass, num_classes=130)).astype(np.float32)
        val_input_chord_root_left = to_categorical(val_input_chord_root_left, num_classes=128)
        val_input_chord_root_right = to_categorical(val_input_chord_root_right, num_classes=128)
        val_input_chord_duration_left = to_categorical(val_input_chord_duration_left, num_classes=20)
        val_input_chord_duration_right = to_categorical(val_input_chord_duration_right, num_classes=20)
        val_input_condition_left = np.concatenate((val_input_beat_left, val_input_fermata_left),axis=-1)
        val_input_condition_right = np.concatenate((val_input_beat_right, val_input_fermata_right), axis=-1)
        val_input_chord_left = np.concatenate((val_input_chord_duration_left, val_input_chord_intervals_left, val_input_chord_root_left), axis=-1)
        val_input_chord_right = np.concatenate((val_input_chord_duration_right, val_input_chord_intervals_right, val_input_chord_root_right), axis=-1)
        '''val_input_condition_left = np.concatenate((val_input_beat_left, val_input_fermata_left),
                                                  axis=-1)
        val_input_condition_right = np.concatenate(
            (val_input_beat_right, val_input_fermata_right), axis=-1)'''
    return (input_melody_left, input_melody_right, input_condition_left, input_condition_right,input_chord_left, input_chord_right,output_alto, output_tenor, output_bass), \
        (val_input_melody_left, val_input_melody_right, val_input_condition_left, val_input_condition_right,val_input_chord_left,val_input_chord_right,
         val_output_alto, val_output_tenor, val_output_bass),filenames
 

 
# 构造mutil head attention层
class MutilHeadAttention_pitch(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        # d_model 必须可以正确分为各个头
        assert d_model % num_heads == 0
        # 分头后的维度
        self.depth = d_model // num_heads
        
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        
        self.dense = tf.keras.layers.Dense(d_model)
        
    def split_heads(self, x, batch_size):
        # 分头, 将头个数的维度 放到 seq_len 前面
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])
    
    def call(self, v, k=None, q=None, mask=None):
        if k is None:
            k = v
            q = v
        batch_size = tf.shape(q)[0]
        
        # 分头前的前向网络，获取q、k、v语义
        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)
        v = self.wv(v)
        
        # 分头
        q = self.split_heads(q, batch_size) # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        # scaled_attention.shape == (batch_size, num_heads, seq_len_v, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        
        # 通过缩放点积注意力层
        scaled_attention, attention_weights = scaled_dot_product_attention(
        q, k, v, mask)
        # 把多头维度后移
        scaled_attention = tf.transpose(scaled_attention, [0, 2, 1, 3]) # (batch_size, seq_len_v, num_heads, depth)
 
        # 合并多头
        concat_attention = tf.reshape(scaled_attention, 
                                      (batch_size, -1, self.d_model))
        
        # 全连接重塑
        output = self.dense(concat_attention)
        return output
 
        
        
    
 
import tensorflow as tf
from tensorflow.keras import layers
 
def scaled_dot_product_attention(q, k, v, mask):
    # query key 相乘获取匹配关系
    matmul_qk = tf.matmul(q, k, transpose_b=True)
    
    # 使用dk进行缩放
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)
    
    # 掩码
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)
        
    # 通过softmax获取attention权重
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    
    # attention 乘上value
    output = tf.matmul(attention_weights, v) # （.., seq_len_v, depth）
    
    return output, attention_weights
    
 
class Autoencoder(Model):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            #layers.Flatten(),
            layers.Dense(130*3, activation='relu'),
            BatchNormalization() ,
            layers.Dense(256, activation='relu'),
            BatchNormalization() ,

            layers.Dense(128, activation='relu'),
            BatchNormalization() ,

            layers.Dense(64, activation='relu'),
            BatchNormalization() ,
            layers.Dense(32, activation='sigmoid')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            BatchNormalization() ,
            layers.Dense(128, activation='relu'),
            BatchNormalization() ,
            layers.Dense(256, activation='relu'),
            BatchNormalization() ,
            layers.Dense(130*3, activation='sigmoid')
        ])


    def call(self, x,decode =False):
        
        if decode:
            decoded = self.decoder(x)
            decoded = tf.reshape(decoded, (tf.shape(x)[0],320,130,3))
            return decoded
        else:
            
            x = tf.reshape(x,(tf.shape(x)[0],320,130*3))
            encoded = self.encoder(x)
            decoded = self.decoder(encoded)
            decoded = tf.reshape(decoded, (tf.shape(x)[0],320,130,3))
            return decoded, encoded
    
    def get_config(self):
        config = {}
        return config
    
    
    
    
class get_weight(tf.keras.layers.Layer):
    def __init__(self, size):
        self.learnable_vector = tf.Variable(lambda :tf.random.normal(shape=(size,)), trainable=True,name='input_weights')
 
    def __call__(self):
        return self.learnable_vector
 
 
class Dense_Attention_Block(tf.keras.layers.Layer):
    def __init__(self, num_layers = NUM_LAYERS, num_heads = NUM_HEADS, key_size = SEGMENT_LENGTH, query_size = SEGMENT_LENGTH, value_size = SEGMENT_LENGTH, num_hiddens = RNN_SIZE, dropout=DROPOUT,mode='add',name='dense_att'):
        super(Dense_Attention_Block, self).__init__(name=name)
 
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_size = key_size
        self.query_size = query_size
        self.value_size = value_size
        self.num_hiddens = num_hiddens
        self.dropout = dropout
        self.mode = mode
        self.d_in = num_hiddens*4
        self.attention_layers = [MutilHeadAttention_pitch(d_model=num_hiddens,num_heads=self.num_heads) for _ in range(num_layers)]
        self.dense_layers1 = [Dense(self.d_in) for _ in range(num_layers)]
        self.dense_layers2 = [Dense(num_hiddens) for _ in range(num_layers)]
        self.dense_connection =  [Dense(num_hiddens) for _ in range(num_layers)]
        # self.input_weight = tf.Variable(lambda :tf.random.normal(shape=(2,)), trainable=True,name=name+'x&y_weights')
        self.batch_norms1 = [LayerNormalization() for _ in range(num_layers)]
        self.batch_norms2 = [LayerNormalization() for _ in range(num_layers)]
    def call(self, inputs,noise = None):
        
        if noise is not None:
            inputs = concatenate([inputs,noise])
        merge = inputs
        x=[inputs]
        # 迭代应用多层注意力块
        for i in range(self.num_layers):
            attention_output = self.attention_layers[i](merge)
            merge = tf.keras.layers.Add()([attention_output, merge])
            
            merge = self.batch_norms1[i](merge)
            
            
            dense_output = self.dense_layers1[i](merge)
            dense_output = tensorflow.nn.relu(dense_output)
            dense_output = self.dense_layers2[i](dense_output)
            merge = tf.keras.layers.Add()([dense_output, merge])
            merge = self.batch_norms2[i](merge)
            #merge = tensorflow.nn.leaky_relu(merge)
            
            x.append(merge)
            for j in range(len(x)-1):
                merge = concatenate([merge, x[j]])
            merge = self.dense_connection[i](merge)
            
        return merge
 
    def get_config(self):
        config = super(Dense_Attention_Block, self).get_config()
        config.update({
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'key_size': self.key_size,
            'query_size': self.query_size,
            'value_size': self.value_size,
            'num_hiddens': self.num_hiddens,
            'dropout': self.dropout
        })
        return config
    
    
class Condition_merge_Block(tf.keras.layers.Layer):
    def __init__(self, num_layers = NUM_LAYERS, num_heads = NUM_HEADS, key_size = SEGMENT_LENGTH, query_size = SEGMENT_LENGTH, value_size = SEGMENT_LENGTH, num_hiddens = RNN_SIZE, dropout=DROPOUT,mode='add',name='dense_att'):
        super(Condition_merge_Block, self).__init__(name=name)
 
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_size = key_size
        self.query_size = query_size
        self.value_size = value_size
        self.num_hiddens = num_hiddens
        self.dropout = dropout
        self.mode = mode
        self.d_in = num_hiddens
        self.attention_layer = MutilHeadAttention_pitch(d_model=num_hiddens,num_heads=self.num_heads)
        self.cross_att = MutilHeadAttention_pitch(d_model=num_hiddens,num_heads=self.num_heads)
        self.dense_layers1 = Dense(self.d_in) 
        self.dense_layers2 = Dense(num_hiddens)
        # self.input_weight = tf.Variable(lambda :tf.random.normal(shape=(2,)), trainable=True,name=name+'x&y_weights')
        self.batch_norm1 = LayerNormalization()
        self.batch_norm2 = LayerNormalization() 
        self.batch_norm3 = LayerNormalization() 
    def call(self,condition,noise = None):
        
        
        x = noise
        
        attention_output = self.attention_layer(x)
        x = tf.keras.layers.Add()([attention_output, x])
        x = self.batch_norm1(x)
        
        x = self.cross_att(condition,condition,x)
        x = self.batch_norm2(x)
        
        
        dense_output = self.dense_layers1(x)
        dense_output = tensorflow.nn.relu(dense_output)
        dense_output = self.dense_layers2(dense_output)
        x = tf.keras.layers.Add()([dense_output, x])
        x = self.batch_norm3(x)
        
 
        return x
 
    def get_config(self):
        config = super(Condition_merge_Block, self).get_config()
        config.update({
        })
        return config    

import tensorflow as tf
import numpy as np
 
 
def custom_loss(y_true, y_pred):
    # 将目标中为0的位置的损失设为0
    mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
    loss = K.mean(K.square(y_true - y_pred) * mask, axis=-1)
    return loss
    
class SinusoidalPosEmb(tf.keras.layers.Layer):
    def __init__(self, dim,name):
        super(SinusoidalPosEmb, self).__init__(name = name)
        half_dim = dim // 2
        emb = np.exp(np.arange(half_dim) * - np.log(10000) / (half_dim - 1))
        self.emb = tf.constant(emb, dtype=tf.float32)
        
    def call(self, x):
        emb = x[:, tf.newaxis] * self.emb[tf.newaxis, :]
        emb = tf.concat([tf.math.sin(emb), tf.math.cos(emb)], axis=-1)
        return emb
    def get_config(self):
        config = {
            'emb': self.emb.shape[-1] * 2,  # Save the full dimension
            'name': self.name,
        }
        base_config = super(SinusoidalPosEmb, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def build_my_diffmodel(rnn_size=RNN_SIZE, num_layers=NUM_LAYERS, seg_length=320, dropout=DROPOUT, weights_path=None,codelength=32,
                training=False):
    input_melody_left = Input(shape=(seg_length, 130), name='input_melody_left')
    
    input_condition_left = Input(shape=(seg_length, 6), name='input_condition_left')
 
    input_chord_left = Input(shape=(seg_length, 160), name='input_chord_left')
 
    # Output shift embedding
 
    input_noise = Input(shape=(seg_length,codelength), name='input_noise')
    '''input_tenor_noise = Input(shape=(seg_length,codelength), name='input_tenor_noise')
    input_bass_noise = Input(shape=(seg_length,codelength), name='input_bass_noise')'''
    time = Input(shape=(1,), name='time')
    t = Dense(int(seg_length/2), activation='relu',name='time_embedding1')(time)
    emb = SinusoidalPosEmb(dim = seg_length, name='sinusoidal_pos_embedding')
    t = emb(t)
    t = tf.transpose(t, perm=[0,2,1])
    t = tf.tile(t, [1,1,rnn_size])
 
    
    
    
    melody_left = TimeDistributed(Dense(rnn_size, activation='relu'), name='melody_left_embedding')(input_melody_left)
 
    condition_left = TimeDistributed(Dense(rnn_size, activation='relu'), name='condition_left_embedding')(
        input_condition_left)
 
    chord_left = TimeDistributed(Dense(rnn_size, activation='relu'), name='chord_left_embedding')(
        input_chord_left)
 
    
    all_condition = tf.concat([melody_left,condition_left,chord_left],axis=-1)



 
 
    noise = Dense(rnn_size , activation='relu',
                                              name='nosie_embedding_channel')(input_noise)
 
 
    dense_att_l = Dense_Attention_Block(num_hiddens=rnn_size*4,name='dense_att_l')
    
    dense_att_noise = Dense_Attention_Block(name='dense_att_noise')
 
    all_condition = concatenate([all_condition,t])
    all_condition = dense_att_l(all_condition)
    
    all_condition = Dense(rnn_size, activation='relu',name='all_condition_dense1')(all_condition)
    
    noise= dense_att_noise(noise,noise =None)
    
    condition_merge_block = Condition_merge_Block(num_hiddens=rnn_size,name='condition_merge_block')
    
    merge = condition_merge_block(all_condition,noise)
    #merge = concatenate([all_condition,noise,t])
    
    
    #merge = LSTM(rnn_size*6,name='merge_lstm1' ,return_sequences=True)(merge)
    attention_blocks = Dense_Attention_Block(num_hiddens=RNN_SIZE, mode='add', name='dense_att_merge')
    merge = attention_blocks(merge)

 
    target = Dense(32,activation='sigmoid',name='target')(merge)
    
    model = Model(inputs=[input_melody_left,  input_condition_left,input_chord_left, input_noise, time]
            ,
            outputs=[target]
        )
    model.compile(optimizer='adamax',
                  loss=custom_loss,
                  metrics=['accuracy'],
                  )
    
    if weights_path == None:
        model.summary()
 
    else:
        print('load weights')
        model.load_weights(weights_path)
    model.summary()
 
    return model


class CustomCallback(Callback):
    def __init__(self):
        super(CustomCallback, self).__init__()
        self.t = [] 
        self.x_t = []
        self.noise = []
        self.t_val = [] 
        self.x_t_val = []
        self.noise_val = []
        self.x_encoded = []
    def on_epoch_begin(self, epoch, logs=None):
        self.t = [] 
        self.x_t = []
        self.noise = []
        self.t_val = [] 
        self.x_t_val = []
        self.noise_val = []
        self.x_encoded_val = []
        autoencoder = Autoencoder()

        load_tensor = tf.ones((1, 320,3,130))
        out1,out2 = autoencoder(load_tensor)
        autoencoder.load_weights('autodencoder_dense_crosssig.hdf5')
        # 冻结模型的权重
        for layer in autoencoder.layers:
            layer.trainable = False

        diff = GaussianDiffusion(image_shape=(data[0].shape[0],320, 3,130), timesteps=100, objective='ddim', eta=1.0,
                             beta_schedule='linear')
        t = diff.sample_timesteps(n=data[0].shape[0])
        x_decoded,x_encoded = autoencoder(tf.stack(data[-3:], axis=2))
        #x_encoded = tf.transpose(x_encoded, [0, 2, 1])
        x_t, noise = diff.noise_images(x_encoded, t)#(None,320,32)
        
        t = tf.expand_dims(t,-1)
        self.noise.append(noise)
        self.t.append(t)
        self.x_t.append(x_t)
        self.x_encoded .append(x_encoded)
        t_val = diff.sample_timesteps(n=data_val[0].shape[0])
        x_decoded_val, x_encoded_val = autoencoder(tf.stack(data_val[-3:], axis=2))
        #x_encoded_val = tf.transpose(x_encoded_val, [0, 2, 1])
        x_t_val, noise_val = diff.noise_images(x_encoded_val, t_val)
        t_val = tf.expand_dims(t_val,-1)
        self.noise_val.append(noise_val)
        self.t_val.append(t_val)
        self.x_t_val.append(x_t_val)
        self.x_encoded_val.append(x_encoded_val)
        
    def on_epoch_end(self, epoch, logs=None):
        self.t.clear()
        self.x_t.clear()
        self.noise.clear()
        self.t_val.clear()
        self.x_t_val.clear()
        self.noise_val.clear()
        self.x_encoded.clear()
    
    def get_config(self):
        config = {
            't': self.t,
            'x_t': self.x_t,
            'noise': self.noise,
            't_val': self.t_val,
            'x_t_val': self.x_t_val,
            'noise_val': self.noise_val,
        }
        base_config = super(CustomCallback, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
 
from tensorflow.keras.callbacks import ModelCheckpoint, LambdaCallback
from networks import GaussianDiffusion
 
class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
        self.writer = tf.summary.create_file_writer(log_dir)

    def on_train_batch_end(self, batch, logs=None):
        logs = logs or {}
        with self.writer.as_default():
            for name, value in logs.items():
                tf.summary.scalar(name, value, step=batch)
            self.writer.flush()

def train_model(data,
                data_val,
                batch_size=BATCH_SIZE,
                epochs=EPOCHS,
                verbose=1,
                weights_path=WEIGHTS_PATH):
    model = build_my_diffmodel(training=True)
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print("checkpoint loaded")
        except:
            os.remove(weights_path)
            print("checkpoint deleted")
    val = VAL
    # Set monitoring indicator
    if len(data_val[0]) != 0:
        monitor = 'val_loss'
    else:
        monitor = 'loss'

    # Save weights
    checkpoint = ModelCheckpoint(filepath=weights_path,
                                 monitor=monitor,
                                 verbose=0,
                                 save_best_only=True,
                                 mode='min',
                                 save_freq='epoch')

    final_checkpoint = ModelCheckpoint(filepath='last'+weights_path, 
                                       save_best_only=False)

    custom_tensorboard_callback = CustomTensorBoard(log_dir='./tf_dir', histogram_freq=1, write_graph=False, write_images=False)

    custom_callback = CustomCallback()
    custom_callback.on_epoch_begin(epoch=1)

    if not PRE_NOISE:
        if val:
            # With validation set
            history = model.fit(x={'input_melody_left': data[0],
                                   'input_condition_left': data[2],
                                   "input_chord_left": data[4],
                                   'input_noise': custom_callback.x_t[-1],
                                   'time': custom_callback.t[-1]
                                   },
                                y={'target': custom_callback.x_encoded[-1]},
                                validation_data=({'input_melody_left': data_val[0],
                                                  'input_condition_left': data_val[2],
                                                  "input_chord_left": data_val[4],
                                                  'input_noise': custom_callback.x_t_val[-1],
                                                  'time': custom_callback.t_val[-1]
                                                  },
                                                 {'target': custom_callback.x_encoded_val[-1]}),
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=verbose,
                                callbacks=[checkpoint, custom_tensorboard_callback, custom_callback, final_checkpoint])
        else:
            # Without validation set
            history = model.fit(x={'input_melody_left': data[0],
                                   'input_condition_left': data[2],
                                   "input_chord_left": data[4],
                                   'input_alto_noise': custom_callback.x_t[-1][:,:,0],
                                   'input_tenor_noise': custom_callback.x_t[-1][:,:,1],
                                   'input_bass_noise': custom_callback.x_t[-1][:,:,2],
                                   'time': custom_callback.t[-1]
                                   },
                                y={'target_alto': data[6],
                                   'target_tenor': data[7],
                                   'target_bass': data[8]},
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=verbose,
                                callbacks=[checkpoint, custom_tensorboard_callback, custom_callback, final_checkpoint])
    else:
        if val:
            # With validation set
            history = model.fit(x={'input_melody_left': data[0],
                                   'input_condition_left': data[2],
                                   "input_chord_left": data[4],
                                   'input_noise': custom_callback.x_t[-1],
                                   'time': custom_callback.t[-1]
                                   },
                                y={'target': custom_callback.noise[-1]},
                                validation_data=({'input_melody_left': data_val[0],
                                                  'input_condition_left': data_val[2],
                                                  "input_chord_left": data_val[4],
                                                  'input_noise': custom_callback.x_t_val[-1],
                                                  'time': custom_callback.t_val[-1]
                                                  },
                                                 {'target': custom_callback.noise_val[-1]}),
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=verbose,
                                callbacks=[checkpoint, custom_tensorboard_callback, custom_callback, final_checkpoint])
        else:
            # Without validation set
            history = model.fit(x={'input_melody_left': data[0],
                                   'input_condition_left': data[2],
                                   "input_chord_left": data[4],
                                   'input_alto_noise': custom_callback.x_t[-1][:,:,0],
                                   'input_tenor_noise': custom_callback.x_t[-1][:,:,1],
                                   'input_bass_noise': custom_callback.x_t[-1][:,:,2],
                                   'time': custom_callback.t[-1]
                                   },
                                y={'target_alto': custom_callback.noise[-1][:,:,0],
                                   'target_tenor': custom_callback.noise[-1][:,:,1],
                                   'target_bass': custom_callback.noise[-1][:,:,2]},
                                batch_size=batch_size,
                                epochs=epochs,
                                verbose=verbose,
                                callbacks=[checkpoint, custom_tensorboard_callback, custom_callback])
    return history
 
 
 
 
if __name__ == "__main__":
    # Load the training and validation sets
    data, data_val,filenames = create_training_data()
 
    #data = create_training_data1()
    # Train model
    history = train_model(data, data_val)