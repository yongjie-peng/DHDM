#from layers import *
from functools import partial
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras import Sequential
import tensorflow.keras.layers as nn
import tensorflow_addons as tfa
import math
from tensorflow import einsum
from einops import rearrange
import numpy as np
from einops.layers.tensorflow import Rearrange

def linear_beta_schedule(timesteps):
    scale = 1
    beta_start = scale * 0.0001
    beta_end = scale * 0.02

    return tf.cast(tf.linspace(beta_start, beta_end, timesteps), tf.float32)

def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = tf.cast(tf.linspace(0, timesteps, steps), tf.float32)

    alphas_cumprod = tf.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])

    return tf.clip_by_value(betas, 0, 0.999)



def extract(x, t):
    return tf.gather(x, t)[:,None, None]
    #return tf.gather(x, t)[:,None, None]

class GaussianDiffusion(Model):
    def __init__(self, image_shape, timesteps=1000, objective='ddpm', eta=1.0, beta_schedule='cosine'):
        super(GaussianDiffusion, self).__init__()

        # 初始化模型的参数
        self.image_shape = image_shape
        self.timesteps = timesteps
        self.objective = objective

        # 根据 beta_schedule 选择不同的 beta 衰减调度
        if beta_schedule == 'linear':
            self.beta = linear_beta_schedule(self.timesteps)
        elif beta_schedule == 'cosine':
            self.beta = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        # 计算 alpha、alpha_hat、alpha_hat_prev 和 eta 的值
        self.alpha = 1 - self.beta
        self.alpha_hat = tf.math.cumprod(self.alpha, axis=0)
        self.alpha_hat_prev = tf.pad(self.alpha_hat[:-1], paddings=[[1, 0]], constant_values=1)

        # 根据 objective 类型设置 eta 的值
        if self.objective == 'ddim':
            self.eta = 0.0
        elif self.objective == 'ddpm':
            self.eta = 1.0
        else:  # general form
            self.eta = eta

    def sample_timesteps(self, n):
        return tf.random.uniform(shape=[n], minval=0, maxval=self.timesteps, dtype=tf.int32)

    def noise_images(self, x, t):  # forward process q
        
        sqrt_alpha_hat = tf.sqrt(extract(self.alpha_hat, t))
        sqrt_one_minus_alpha_hat = tf.sqrt(1 - extract(self.alpha_hat, t))

        eps = tf.random.normal(shape=x.shape)

        x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * eps
        
        return x, eps

    def sample(self, model, n, melody_left, melody_right, condition_left, condition_right,chord_left,chord_right, alto, tenor, bass, start_x=None, start_t=None):  # reverse process p
        # 初始化或使用给定的起始图像
        if start_x is None:
            '''
            alto_mean = 19.42
            alto_std = 32.82
            tenor_mean = 28.773
            tenor_std = 49.22
            bass_mean = 33.65
            bass_std =51.62
            x_alto = tf.random.normal(shape=[n] + [130],mean=alto_mean,stddev=alto_std)
            x_tenor = tf.random.normal(shape=[n] + [130],mean=tenor_mean,stddev=tenor_std)
            x_bass = tf.random.normal(shape=[n] + [130],mean=bass_mean,stddev=bass_std)
            x = tf.stack([x_alto,x_tenor,x_bass],axis=-1)
            '''
            x = tf.random.normal(shape=[n] + [130,3])
            
        else:
            x = start_x
        ratio = 0.032
        '''noisy_intial = np.random.binomial(1, x*(1-beta)+ratio*beta)
        noisy = np.copy(noisy_intial)'''
        noisy_intial = np.copy(x)
        #noisy = np.copy(noisy_intial)
        # 逆向迭代生成样本
        x_tlist = []
        for i in tqdm(reversed(range(1, self.timesteps if start_t is None else start_t[0])), desc='sampling loop time step', total=self.timesteps):
            #print(i)
            t = tf.ones(n, dtype=tf.int32) * i
            inputs = {'input_melody_left': melody_left,
                           'input_melody_right': melody_right,
                           'input_condition_left': condition_left,
                           'input_condition_right': condition_right,
                           "input_chord_left": chord_left,
                           'input_chord_right': chord_right,
                           'input_alto': alto,
                            'input_tenor': tenor,
                            'input_bass': bass,
                           'input_alto_noise': x[:, :, 0],
                           'input_tenor_noise': x[:, :, 1],
                           'input_bass_noise': x[:, :, 2],
                           'time': t
                           }
            predictions = tf.stack(model(inputs, training=False),axis=-1)
            predicted_noise = x-predictions
            # 提取 alpha、alpha_hat、beta 的值
            alpha = extract(self.alpha, t)
            alpha_hat = extract(self.alpha_hat, t)
            beta = extract(self.beta, t)

            alpha_hat_prev = extract(self.alpha_hat_prev, t)
            beta_hat = beta * (1 - alpha_hat_prev) / (1 - alpha_hat)  # 类似于 beta
            if i > 1:
                noise = tf.random.normal(shape=x.shape)
            else: # last step
                noise = tf.zeros_like(x)
            # 简化操作，确保噪声的形状与输入图像相匹配
            #noise = tf.random.normal(shape=x.shape)

            if self.objective == 'ddpm':
                # 根据 DDPM 的生成方式计算样本
                #try1 = (beta / (tf.sqrt(1 - alpha_hat))) * predicted_noise
                #direction_point = 1 / tf.sqrt(alpha) * (x - (beta / (tf.sqrt(1 - alpha_hat))) * predicted_noise)
                direction_point = 1 / tf.sqrt(alpha) * (x - (beta / (tf.sqrt(1 - alpha_hat))) * (x-predicted_noise))
                random_noise = beta_hat * noise
                x = direction_point + random_noise
            elif self.objective == 'ddim':
                sigma = 0.0
                predict_x0 = alpha_hat_prev * (x - tf.sqrt(1 - alpha_hat) * predicted_noise) / tf.sqrt(alpha_hat)
                #predict_x0 = alpha_hat_prev * (x - tf.sqrt(1 - alpha_hat) * (x-predicted_noise)) / tf.sqrt(alpha_hat)
                direction_point = tf.sqrt(1 - alpha_hat_prev - tf.square(sigma)) * predicted_noise
                #direction_point = tf.sqrt(1 - alpha_hat_prev - tf.square(sigma)) * (x-predicted_noise)
                random_noise = sigma * noise

                x = predict_x0 + direction_point + random_noise

            elif self.objective == 'lms':
                sigma = 0.0
                predict_x0 =  alpha_hat_prev * predictions / tf.sqrt(alpha_hat)
                direction_point = tf.sqrt(1 - alpha_hat_prev - tf.square(sigma)) * predicted_noise
                random_noise = sigma * noise

                x = predict_x0 + direction_point + random_noise

            elif self.objective == 'dds':
                threshold = 0.3
                #predict_x0 = predicted_noise
                predicted_noise = predicted_noise >= threshold

                beta = (self.timesteps-i)/self.timesteps

                delta = np.array(predicted_noise ^ noisy_intial)
                mask = np.random.binomial(1,  (delta.astype(np.int32)) * beta)
                #x = predicted_noise*(1-mask) + noisy_intial * mask
                x= tf.cast(predicted_noise, dtype=tf.float32) * (1 - tf.cast(mask, dtype=tf.float32)) + tf.cast(noisy_intial, dtype=tf.float32) * tf.cast(mask, dtype=tf.float32)

            else: # general form
                sigma = self.eta * tf.sqrt((1 - alpha_hat_prev) / (1 - alpha_hat)) * tf.sqrt(1 - (alpha_hat / alpha_hat_prev))
                predict_x0 = alpha_hat_prev * (x - tf.sqrt(1 - alpha_hat) * predicted_noise) / tf.sqrt(alpha_hat)
                direction_point = tf.sqrt(1 - alpha_hat_prev - tf.square(sigma)) * predicted_noise
                random_noise = sigma * noise

                x = predict_x0 + direction_point + random_noise
            #x_tlist.append(x)
        return x

    def sample_from_timestep(self, model, image, timesteps):
        # 生成给定时间步的噪声图像和去噪图像
        denoised_images = []
        noised_images = []

        for t in timesteps:
            noised_image, _ = self.noise_images(image, [t])
            denoised_image = self.sample(model, n=noised_image.shape[0], start_x=noised_image, start_t=t)
            denoised_images.append(denoised_image)
            noised_images.append(noised_image)

        # 将噪声图像和去噪图像拼接在一起返回
        x = tf.concat(noised_images + denoised_images, axis=0)
        return x

