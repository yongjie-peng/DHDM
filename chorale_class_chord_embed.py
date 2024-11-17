import os
import numpy as np
from config import *
from music21 import *
from tqdm import trange
#from model_notot_diff_huigui import build_my_diffmodel
from HDM_class_chord_embed import build_my_diffmodel
from load_old import get_filenames, convert_files
from tensorflow.python.keras.utils.np_utils import to_categorical
import tensorflow as tf
# use cpu
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from vqgan import Autoencoder
import scipy
import math
from numpy import cov
from numpy import trace
from numpy import iscomplexobj
from numpy.random import random
from scipy.linalg import sqrtm
from scipy.special import erf
from sklearn import metrics

def calculate_oa(mu1, sigma1, mu2, sigma2):
    # 计算交点c
    c = (sigma2**2 * mu1 + sigma1**2 * mu2) / (sigma1**2 + sigma2**2)
    
    # 计算OA
    '''if sigma1 == 0 or sigma2 == 0:
        print("Error: Variance cannot be zero.")
        sigma = max(sigma1, sigma2)
        sigma1 = sigma
        sigma2 = sigma'''
    oa = 1 - erf((c - mu1) / (np.sqrt(2) * sigma1)) + erf((c - mu2) / (np.sqrt(2) * sigma2))
    return oa

def calculate_consistency_variance(oa_values, ground_truth):
    # 计算μOA和σ2OA
    mu_oa = np.mean(oa_values)
    sigma2_oa = np.var(oa_values)
    
    # 计算μGT和σ2GT
    mu_gt = np.mean(ground_truth)
    sigma2_gt = np.var(ground_truth)
    
    # 计算一致性和方差
    consistency = max(0, 1 - abs(mu_oa - mu_gt) / mu_gt)
    variance = max(0, 1 - abs(sigma2_oa - sigma2_gt) / sigma2_gt)
    
    return consistency, variance


def sliding_window_OA(sequence, window_length=64, stride=32):
    
    oa_list = []
    num_windows = (len(sequence) - window_length) // stride + 1

    for i in range(num_windows):
        window_data = sequence[i * stride: i * stride + window_length]
        next_window_data = sequence[(i + 1) * stride: (i + 1) * stride + window_length]
        
        # 计算OA
        oa_result = calculate_oa(np.mean(window_data), np.std(window_data)+np.finfo(np.float64).eps, np.mean(next_window_data), np.std(next_window_data)+np.finfo(np.float64).eps)
        oa_list.append(oa_result)
    return oa_list


def get_OA_consistency_variance(generation,groundtruth):
    oa_result = sliding_window_OA(np.array(generation)/130)
    oa_result_groundtruth = sliding_window_OA(np.array(groundtruth)/130)
    #consistency, variance = calculate_consistency_variance([oa_result], np.array(groundtruth)/130)
    consistency, variance = calculate_consistency_variance([oa_result], [oa_result_groundtruth])
    return  consistency, variance
 
def mmd_rbf(real, fake, gamma=1.0):
  """(RBF) kernel distance.
  
  Lower score is better.
  """
  XX = metrics.pairwise.rbf_kernel(real, real, gamma)
  YY = metrics.pairwise.rbf_kernel(fake, fake, gamma)
  XY = metrics.pairwise.rbf_kernel(real, fake, gamma)
  return XX.mean() + YY.mean() - 2 * XY.mean()


def mmd_polynomial(real, fake, degree=2, gamma=1, coef0=0):
  """(Polynomial) kernel distance.
  
  Lower score is better.
  """
  XX = metrics.pairwise.polynomial_kernel(real, real, degree, gamma, coef0)
  YY = metrics.pairwise.polynomial_kernel(fake, fake, degree, gamma, coef0)
  XY = metrics.pairwise.polynomial_kernel(real, fake, degree, gamma, coef0)
  return XX.mean() + YY.mean() - 2 * XY.mean()
 
def frechet_distance(real, fake):
  """Frechet distance.
  
  Lower score is better.
  """
  mu1, sigma1 = np.mean(real, axis=0), np.cov(real, rowvar=False)
  mu2, sigma2 = np.mean(fake, axis=0), np.cov(fake, rowvar=False)
  diff = mu1 - mu2
  covmean, _ = scipy.linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  eps = np.finfo(np.float64).eps
  if not np.isfinite(covmean).all():
    msg = ('fid calculation produces singular product; '
           'adding %s to diagonal of cov estimates') % eps
    print(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = scipy.linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # Numerical error might give slight imaginary component
  if np.iscomplexobj(covmean):
    '''if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError('Imaginary component {}'.format(m))'''
    covmean = covmean.real

  assert np.isfinite(covmean).all() and not np.iscomplexobj(covmean)

  tr_covmean = np.trace(covmean)

  frechet_dist = diff.dot(diff)
  frechet_dist += np.trace(sigma1) + np.trace(sigma2)
  frechet_dist -= 2 * tr_covmean
  return frechet_dist
 
 
def calculate_fid(act1, act2):
    # calculate mean and covariance statistics
    mu1, sigma1 = act1.mean(axis= 0), cov(act1, rowvar= False)
    mu2, sigma2 = act2.mean(axis= 0), cov(act2, rowvar= False)
 
    # calculate sum squared difference between means
    ssdiff = np.sum((mu1 - mu2)* 2.0)
 
    # calculate sqrt of product between cov
    covmean = sqrtm(sigma1.dot(sigma2))
    fid = 0
    # check and correct imaginary numbers from sqrt
    if iscomplexobj(covmean):
        covmean = covmean.real
        # calculate score
        fid = ssdiff + trace(sigma1 + sigma2 - 2.0*covmean)
    return fid

def get_chord_tones(chord_vec):
    chord_tones = []

    for idx, n in enumerate(chord_vec):
        if n == 0:
            continue
        k = 0
        while k * 12 + idx < 128:
            chord_tones.append(k * 12 + idx)
            k += 1

    return chord_tones
import tensorflow.keras.backend as K
def custom_loss(y_true, y_pred):
    # 将目标中为0的位置的损失设为0
    mask = K.cast(K.not_equal(y_true, 0), dtype='float32')
    loss = K.mean(K.square(y_true - y_pred) * mask, axis=-1)
    return np.mean(loss)

def chorale_generator(input_melody, input_beat, input_fermata,input_chord_root,model, gap,diff,input_alto,input_tenor,input_bass, seg_length=SEGMENT_LENGTH,
                      chord_gamma=1 - HARMONICITY, onset_gamma=HOMOPHONICITY):
    
    melody_left =   input_melody
    melody_right =   input_melody[::-1]
    beat_left =   input_beat
    beat_right =   input_beat[::-1]
    fermata_left =   input_fermata  
    fermata_right = input_fermata[::-1]
    chord_left =   input_chord_root  
    chord_right =   input_chord_root[::-1]

    input_alto = input_alto 
    input_tenor = input_tenor 
    input_bass = input_bass 

        # one-hot
    melody_left = to_categorical(melody_left, num_classes=130)
    melody_right = to_categorical(melody_right, num_classes=130)
    beat_left = to_categorical(beat_left, num_classes=4)
    beat_right = to_categorical(beat_right, num_classes=4)
    fermata_left = to_categorical(fermata_left, num_classes=2)
    fermata_right = to_categorical(fermata_right, num_classes=2)
    alto = np.expand_dims(input_alto, axis=0)
    tenor = np.expand_dims(input_tenor, axis=0)
    bass = np.expand_dims(input_bass, axis=0)
    alto = np.array(to_categorical(alto, num_classes=130)).astype(np.float32)
    tenor = np.array(to_categorical(tenor, num_classes=130)).astype(np.float32)
    bass = np.array(to_categorical(bass, num_classes=130)).astype(np.float32)
    
    condition_left = np.concatenate((beat_left, fermata_left), axis=-1)
    condition_right = np.concatenate((beat_right, fermata_right), axis=-1)
    chord_left = np.array(chord_left).astype(np.float32)
    chord_right = np.array(chord_right).astype(np.float32)

    # expand one dimension
    melody_left = np.expand_dims(melody_left, axis=0)
    melody_right = np.expand_dims(melody_right, axis=0)
    condition_left = np.expand_dims(condition_left, axis=0)
    condition_right = np.expand_dims(condition_right, axis=0)
    chord_left = np.expand_dims(chord_left, axis=0)
    chord_right = np.expand_dims(chord_right, axis=0)
    t = diff.sample_timesteps(n=1)
    autoencoder = Autoencoder()
    load_tensor = tf.ones((1,  320,3,130))
    autoencoder(load_tensor)

    autoencoder.load_weights(VAE_PATH)
    output_decoder,output_encoder = autoencoder(tf.stack([alto,tenor,bass], axis=2),
                                 decode=False)
    #output_encoder = tf.transpose(output_encoder, perm=[0, 2, 1])
    x_t, noise = diff.noise_images(output_encoder, t)

    #output_model = model.predict([melody_left, melody_right, condition_left, condition_right,chord_left,chord_left,x_t[:, :, 0],x_t[:, :, 1],x_t[:, :, 2],t])
    #predictions = model([melody_left, melody_right, condition_left, condition_right,chord_left,chord_left,alto,tenor,bass,t])
    predictions_list = diff.sample(model,1,melody_left, melody_right, condition_left, condition_right,chord_left,chord_right,start_x=x_t,start_t=t)
    # predictions_list = diff.sample(model,1,melody_left, melody_right, condition_left, condition_right,chord_left,chord_right)

    #predictions = output_model[:3]
    #atten_score = output_model[3:5]
    get_pre = []
    loss = []
    for predictions in predictions_list:
        
        #loss .append(custom_loss(tf.stack((alto,tenor,bass), axis=-1),predictions))
        '''song_alto = np.clip(np.round(predictions[:, :, 0]), 0, 129)
        song_tenor = np.clip(np.round(predictions[:, :, 1]), 0, 129)
        song_bass = np.clip(np.round(predictions[:, :, 2]), 0, 129)'''
        song_alto = predictions[:, :, 0]
        song_tenor = predictions[:, :, 1]
        song_bass = predictions[:, :, 2]
        '''song_alto = np.where(song_alto > 100, 129, song_alto)
        song_bass = np.where(song_bass > 100, 129, song_bass)
        song_tenor = np.where(song_tenor > 100, 129, song_tenor)'''
        song_alto = np.where(song_alto < 128 + gap, song_alto - gap, song_alto)
        song_tenor = np.where(song_tenor < 128 + gap, song_tenor - gap, song_tenor)
        song_bass = np.where(song_bass < 128 + gap, song_bass - gap, song_bass)



        get_pre.append([song_alto.astype(int), song_tenor.astype(int), song_bass.astype(int)])
        
    return get_pre,loss
    
 
def txt2music(txt, fermata_txt, ks_list, ts_list):
    if len(ts_list) == 0:
        ts_list = [meter.TimeSignature('c')]

    if len(ks_list) == 0:
        ks_list = [key.KeySignature(0)]

    # Initialization
    notes = [ts_list[0], ks_list[0]]
    pre_element = None
    duration = 0.0
    offset = 0.0

    ks_cnt = ts_cnt = 1
    last_note_is_fermata = False
        # Decode text sequences
    for element in txt + [130]:

        if element != 129:

            # Create new note
            if pre_element != None:

                # If is note
                if pre_element < 128:
                    new_note = note.Note(pre_element)

                # If is rest
                elif pre_element == 128:
                    new_note = note.Rest()

                if fermata_txt[int(offset / 0.25)] == 1 and last_note_is_fermata == False:
                    new_note.expressions.append(expressions.Fermata())
                    last_note_is_fermata = True

                elif fermata_txt[int(offset / 0.25)] != 1:
                    last_note_is_fermata = False

                new_note.quarterLength = duration
                new_note.offset = offset
                notes.append(new_note)

            # Updata offset, duration and save the element
            offset += duration
            duration = 0.25
            pre_element = element

            if ks_cnt < len(ks_list) and offset >= ks_list[ks_cnt].offset:
                notes.append(ks_list[ks_cnt])
                ks_cnt += 1

            if ts_cnt < len(ts_list) and offset >= ts_list[ts_cnt].offset:
                notes.append(ts_list[ts_cnt])
                ts_cnt += 1

        else:

            # Updata duration
            duration += 0.25

    return notes


def export_music(melody, chorale_list, fermata_txt, filename,t, keep_chord=KEEP_CHORD):
    ks_list = []
    ts_list = []
    new_melody = []
    filename = os.path.basename(filename)
    filename = '.'.join(filename.split('.')[:-1])

    # Get meta information
    for element in melody.flat:

        if isinstance(element, meter.TimeSignature):
            ts_list.append(element)

        if isinstance(element, key.KeySignature):
            ks_list.append(element)

        if not isinstance(element, harmony.ChordSymbol):
            new_melody.append(element)

    # Compose four parts
    if keep_chord:
        new_score = [melody]

    else:
        new_score = [stream.Part(new_melody)]

    for i in range(3):
        
        new_part = stream.Part(txt2music(chorale_list[i], fermata_txt, ks_list, ts_list))
        new_part = new_part.transpose(interval.Interval(0))
        new_score.append(new_part)

    # Save as mxl
    new_score = stream.Stream(new_score)

    if not os.path.exists(OUTPUTS_PATH):
            os.makedirs(OUTPUTS_PATH)
    new_score.write('mxl', fp=OUTPUTS_PATH + '/' + filename +'_step_'+str(t)+'.mxl')

from network_4_6 import GaussianDiffusion
import pandas as pd


if __name__ == '__main__':

    # Load model
    #model = build_my_diffmodel(weights_path=r"my_model_diffhuigui_batch8_layer3_mergelayer3_rnn_withlstm_withoutweight_epoch100_adam_head4_linear_1_7.hdf5", training=False)
    
    
    weights_path =r"HDM_class_chord_embed.hdf5"
    model = build_my_diffmodel(weights_path=weights_path)
    config = model.get_config()

    get_all = False
    
    #filenames = get_filenames(r"E:\dataset_清洗_splited")
    filenames = get_filenames(r"E:\full_songs\test_ccd_part1")
    #filenames = get_filenames(r"E:\test_data")
    data_corpus = convert_files(filenames, fromDataset=False)
    all_generation = [[],[],[]]
    all_target = [[],[],[]]
    mmd_rbf_score_list = []
    mmd_polynomial_score_list = []
    loss_list = []
    fid_score_list = []
    # Process each score
    for idx in trange(len(data_corpus)):

        chorale_list = []
        fermata_txt = []
        song_data = data_corpus[idx]
        melody_score = song_data[4]
        filename = song_data[6]
        diff = GaussianDiffusion(image_shape=(len(song_data[0][0]),320, 3,130), timesteps=100, objective='ddlm', eta=1.0,
                                        beta_schedule='linear')
        for part_idx in range(len(song_data[0])):
            input_melody = song_data[0][part_idx]
            num = len(input_melody)-input_melody.count(0)
            input_beat = song_data[1][part_idx]
            input_fermata = song_data[2][part_idx]
            input_alto = song_data[7][part_idx]
            input_tenor = song_data[8][part_idx]
            input_bass = song_data[9][part_idx]
            input_chord = song_data[3][part_idx]

            gap = song_data[5][part_idx]
            fermata_txt += input_fermata
            output,loss = chorale_generator(input_melody, input_beat, input_fermata, input_chord, model, gap,diff,input_alto,input_tenor,input_bass)
            chorale_list.append(output)
            '''alto_list = chorale_list[0][-1][0][0].tolist()
            tenor_list = chorale_list[0][-1][1][0].tolist()
            bass_list = chorale_list[0][-1][2][0].tolist()
            export_music(melody_score, [alto_list[:num], tenor_list[:num], bass_list[:num]], fermata_txt, filename)'''
            loss_list.append(loss)
        
        '''alto_list = output[-1][0][0].tolist()
        tenor_list = output[-1][1][0].tolist()
        bass_list = output[-1][2][0].tolist()'''
        
        if get_all:
            for i in range(99): 
                alto_list = output[i][0].tolist()[0]
                tenor_list = output[i][1].tolist()[0]
                bass_list = output[i][2].tolist()[0]
                export_music(melody_score, [alto_list[:num], tenor_list[:num], bass_list[:num]], fermata_txt,filename,t=i)
        else:
            alto_list = output[-1][0].tolist()[0]
            tenor_list = output[-1][1].tolist()[0]
            bass_list = output[-1][2].tolist()[0]
            export_music(melody_score, [alto_list[:num], tenor_list[:num], bass_list[:num]], fermata_txt,filename,t="last")
        #计算FID
        generation_score = np.stack((alto_list[:num],tenor_list[:num],bass_list[:num]),axis=0)
        #令generation_score中的所有数值除以130
        generation_score = generation_score/130
        #将alto_list转为one-hot编码
        #generation_score = np.eye(130)[generation_score]

        origin_score = np.stack((input_alto[:num],input_tenor[:num],input_bass[:num]),axis=0)
        origin_score = origin_score/130
        
        
        fid_score = 0
        #fid_score = calculate_fid(generation_score,origin_score)
        fid_score = frechet_distance(origin_score, generation_score)
        fid_score_list.append(fid_score)
        print('FID:%.3f'% fid_score)
        
        #计算MMD
        mmd_rbf_score = mmd_rbf(real=origin_score, fake=generation_score)
        mmd_rbf_score_list.append(mmd_rbf_score)
        print('MMD_rbf_score:%.3f'% mmd_rbf_score)
        mmd_polynomial_score = mmd_polynomial(real=origin_score, fake=generation_score)
        mmd_polynomial_score_list.append(mmd_polynomial_score)
        print('MMD_polynomial_score:%.3f'% mmd_polynomial_score)
        
        
        fid_score_list.append(fid_score)
        all_generation[0] += alto_list[:num]
        all_generation[1] += tenor_list[:num]
        all_generation[2] += bass_list[:num]
        all_target[0] += input_alto[:num]
        all_target[1] += input_tenor[:num]
        all_target[2] += input_bass[:num]
    #计算正确率
    #alto_acc = sum(all_generation[0]==all_target[0])/len(all_generation[0])
    alto_equal_values = sum(generation_value == target_value for generation_value, target_value in zip(all_generation[0], all_target[0]))
    alto_acc = alto_equal_values / len(all_generation[0])
    print('alto_acc',alto_acc)
    tenor_equal_values = sum(generation_value == target_value for generation_value, target_value in zip(all_generation[1], all_target[1]))
    tenor_acc = tenor_equal_values / len(all_generation[1])
    print('tenor_acc',tenor_acc)
    bass_equal_values = sum(generation_value == target_value for generation_value, target_value in zip(all_generation[2], all_target[2]))
    bass_acc = bass_equal_values / len(all_generation[2])
    print('bass_acc',bass_acc)
    #计算loss
    '''loss = sum(loss_list)/len(loss_list)
    print('average_loss',loss)'''
    consistency_list = []
    variance_list = []
    for i in range(3):
        consistency, variance= get_OA_consistency_variance(all_generation[i],all_target[i])
        consistency_list.append(consistency)
        variance_list.append(variance)
    print('OA consistency_alto:%.3f'% consistency_list[0],'OA variance_alto:%.3f'% variance_list[0])
    print('OA consistency_tenor:%.3f'% consistency_list[1],'OA variance_tenor:%.3f'% variance_list[1])
    print('OA consistency_bass:%.3f'% consistency_list[2],'OA variance_bass:%.3f'% variance_list[2])
    oa_consistency = (consistency_list[0]+consistency_list[1]+consistency_list[2])/3
    oa_variance = (variance_list[0]+variance_list[1]+variance_list[2])/3
    #print('OA consistency:%.3f'% (consistency_list[0]+consistency_list[1]+consistency_list[2])/3, 'OA variance:%.3f'% (variance_list[0]+variance_list[1]+variance_list[2])/3)
    print('MMD_rbf_score:%.3f'% (np.mean(mmd_rbf_score_list)),'MMD_polynomial_score:%.3f'% (np.mean(mmd_polynomial_score_list)))
    print('FID_score:%.3f'% (np.mean(fid_score_list)))
    results_df = pd.DataFrame(columns=['Timestep', 'FID', 'MMD_rbf', 'MMD_polynomial', 'Alto Accuracy', 'Tenor Accuracy', 'Bass Accuracy', 'OA Consistency Alto', 'OA Consistency Tenor', 'OA Consistency Bass', 'OA Variance Alto', 'OA Variance Tenor', 'OA Variance Bass'])
    results_df = results_df.append({
        'FID': np.mean(fid_score_list),
        'MMD_rbf': np.mean(mmd_rbf_score_list),
        'MMD_polynomial':np.mean(mmd_polynomial_score_list),
        'Alto Accuracy': alto_acc,
        'Tenor Accuracy': tenor_acc,
        'Bass Accuracy': bass_acc,
        'OA Consistency Alto': consistency_list[0],
        'OA Consistency Tenor': consistency_list[1],
        'OA Consistency Bass': consistency_list[2],
        'OA Variance Alto': variance_list[0],
        'OA Variance Tenor': variance_list[1],
        'OA Variance Bass': variance_list[2],
        "oa consistency": oa_consistency,
        "oa variance": oa_variance,
    }, ignore_index=True)

# 保存DataFrame到CSV文件，文件名为model的名称
results_df.to_csv(f"last_{weights_path}_ccd_{SEED}.csv", index=False)