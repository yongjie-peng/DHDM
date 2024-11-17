from music21 import converter, roman
import itertools

def find_all_possible_chord_progressions(folder_path,chord_folder_path):

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(folder_path):
        # 如果文件名以.xml结尾，就处理该文件
        if file_name.endswith('.mxl'):
            # 拼接文件的完整路径
            file_path = os.path.join(folder_path, file_name)

            s = converter.parse(file_path)  # parse your music file into a music21 object
            chord = s.chordify()  # chordify the music object
            print(file_name)
            
            # 检测目标文件夹是否存在，如果不存在则创建
            if not os.path.exists(chord_folder_path):
                os.makedirs(chord_folder_path)
            files_path=os.path.join(chord_folder_path,file_name.replace('.mxl', '_chords.mxl'))
            chord.write('musicxml',files_path)



def calculate_ctnctr_and_pcs(musicxml_file):
    score = converter.parse(musicxml_file)
    parts = score.parts  # 获取乐谱的所有声部

    total_nc = 0  # 和弦音的总数
    total_np = 0  # “正确”的非和弦音的总数
    total_nn = 0  # 非和弦音的总数
    total_consonance_score = 0  # 总的音程和谐度分数
    total_notes_count = 0  # 总音符数量

    for part in parts:
        for measure in part.getElementsByClass('Measure'):
            chords = measure.getElementsByClass('Chord')
            notes = measure.getElementsByClass('Note')

            for chord in chords:
                chord_pitches = [pitch.pitchClass for pitch in chord.pitches]

                for note in notes:
                    if note.isRest:
                        continue  # 忽略休止符

                    total_notes_count += 1

                    if note.pitch.pitchClass in chord_pitches:  # 判断是否为和弦音
                        total_nc += 1
                    else:  # 非和弦音
                        total_nn += 1
                        next_note = note.next('Note')
                        if next_note and next_note.pitch.pitchClass - note.pitch.pitchClass == 2:  # 判断是否为“正确”的非和弦音
                            total_np += 1

                    # 计算音程和谐度分数
                    melody_pitch_class = note.pitch.pitchClass
                    intervals = [(melody_pitch_class - pitch_class) % 12 for pitch_class in chord_pitches]
                    consonance_score = 1 if min(intervals) in {0, 3, 4, 7, 8, 9} else 0  # 0: unison, 3/4: major/minor third, 7: perfect fifth, 8/9: major/minor sixth
                    total_consonance_score += consonance_score

    #ctnctr = (total_nc + total_np) / ((total_nc + total_nn)+np.finfo(np.float32).eps)
    ctnctr = (total_nc + total_np) / (total_nc + total_nn)
    #pcs = total_consonance_score / (total_notes_count+np.finfo(np.float32).eps)
    pcs = total_consonance_score / total_notes_count
    return ctnctr, pcs

import numpy as np
from music21 import roman

def calculate_mctd(musicxml_file):
    score = converter.parse(musicxml_file)
    parts = score.parts  # 获取乐谱的所有声部

    total_mctd = 0  # 总的旋律-和弦音级距离
    total_notes_count = 0  # 总音符数量

    for part in parts:
        for measure in part.getElementsByClass('Measure'):
            chords = measure.getElementsByClass('Chord')
            notes = measure.getElementsByClass('Note')

            for chord in chords:
                chord_pitches = [pitch.pitchClass for pitch in chord.pitches]

                for note in notes:
                    if note.isRest:
                        continue  # 忽略休止符

                    total_notes_count += 1

                    # 计算旋律-和弦音级距离
                    melody_pitch_class = note.pitch.pitchClass
                    mctd = min([(abs(melody_pitch_class - pitch_class) % 12) for pitch_class in chord_pitches])
                    total_mctd += mctd

    avg_mctd = total_mctd / (total_notes_count++np.finfo(np.float32).eps)
    return avg_mctd

def calculate_ctd(musicxml_file):
    score = converter.parse(musicxml_file)
    parts = score.parts  # 获取乐谱的所有声部

    chord_features = []  # 和弦的PCP特征列表

    for part in parts:
        for measure in part.getElementsByClass('Measure'):
            chords = measure.getElementsByClass('Chord')
            for chord in chords:
                # 计算和弦的PCP特征
                chord_pitches = [pitch.pitchClass for pitch in chord.pitches]
                pcp_feature = np.zeros(12)
                for pitch_class in chord_pitches:
                    pcp_feature[pitch_class] = 1
                chord_features.append(pcp_feature)

    ctd_values = []  # 保存每对相邻和弦的CTD值

    # 计算相邻和弦之间的CTD值
    for i in range(1, len(chord_features)):
        chord1_feature = chord_features[i - 1]
        chord2_feature = chord_features[i]
        distance = np.linalg.norm(chord1_feature - chord2_feature)  # 计算欧氏距离
        ctd_values.append(distance)

    ctd = np.mean(ctd_values)  # 计算CTD的平均值
    return ctd

from collections import Counter
import math

def calculate_che(chord_sequence):
    # 创建和弦出现频率直方图
    chord_counts = Counter(chord_sequence)
    total_chords = len(chord_sequence)

    # 计算每个和弦的相对概率
    relative_probabilities = [count / total_chords for count in chord_counts.values()]

    # 计算熵
    entropy = -sum(prob * math.log(prob) for prob in relative_probabilities)

    return entropy


from collections import Counter

def calculate_chord_coverage(file_path):
    # Load MusicXML file
    score = converter.parse(file_path)
    
    # Extract chord symbols from each part
    chord_histogram = {}
    num_parts = len(score.parts)
    
    for part in score.parts:
        for element in part.recurse():
            if 'ChordSymbol' in element.classes:
                chord_label = element.figure
                chord_histogram[chord_label] = chord_histogram.get(chord_label, 0) + 1
    
    # Calculate chord coverage
    chord_coverage = len(chord_histogram)
    
    # Calculate chord coverage per measure
    chord_coverage_per_measure = chord_coverage / num_parts
    
    return chord_coverage_per_measure


# 示例和弦序列
chord_sequence = ["C", "G", "Am", "F", "C", "G", "Am", "F", "C"]
# 示例和弦序列
chord_sequence = ["C", "G", "Am", "F", "C", "G", "Am", "F", "C"]

'''# 调用函数计算CTnCTR和PCS
musicxml_file = r"D:\learn_project\deepchoir-main_secondcode\dataset\019.mxl"# 将文件路径替换为你的MusicXML文件路径
#musicxml_file = r"E:\dataset_清洗_splited\19.mxl"
ctnctr, pcs = calculate_ctnctr_and_pcs(musicxml_file)

mctd = calculate_mctd(musicxml_file)
ctd = calculate_ctd(musicxml_file)

che = calculate_che(chord_sequence)

# 调用函数计算CC
cc = calculate_cc_from_musicxml(musicxml_file)
print("CC:", cc)
print("CHE:", che)
print("CTD:", ctd)
print("MCTD:", mctd)
print("CTnCTR:", ctnctr)
print("PCS:", pcs)'''
import math
from collections import Counter
from music21 import converter, chord

import os
def extract_chords_and_measures_from_musicxml(file_path):
    # Load the MusicXML file
    score = converter.parse(file_path)
    
    # Extract chords and count measures
    chords = []
    measures = score.getElementsByClass('Measure')
    measure_count = len(measures)
    
    for part in score.parts:
        for element in part.flat.notesAndRests:
            if isinstance(element, chord.Chord):
                chords.append(element.root().name)
        # Count measures in each part
        measure_count = max(measure_count, len(part.getElementsByClass('Measure')))
    
    return chords, measure_count


def calculate_histogram(chord_sequence):
    # Count the occurrences of each chord
    chord_counts = Counter(chord_sequence)
    # Create histogram bins
    chords = list(chord_counts.keys())
    counts = list(chord_counts.values())
    return chords, counts

def normalize_histogram(counts):
    total_counts = sum(counts)
    normalized_counts = [count / total_counts for count in counts]
    return normalized_counts

def calculate_entropy(normalized_counts):
    entropy = -sum(p * math.log(p, 2) for p in normalized_counts if p > 0)
    return entropy

def chord_coverage(chords, measure_count):
    coverage = len(chords)
    # Normalize by measure count
    # if measure_count > 0:
    #     coverage /= measure_count
    return coverage

# 定义函数来计算各参数
def calculate_parameters(musicxml_file):
    # 根据需要调用各个函数来计算参数
    ctnctr, pcs = calculate_ctnctr_and_pcs(musicxml_file)
    mctd = calculate_mctd(musicxml_file)
    ctd = calculate_ctd(musicxml_file)
    # che = calculate_che(chord_sequence)
    # cc = calculate_chord_coverage(musicxml_file)
    # Extract chords from MusicXML file
    chord_sequence, measure_count = extract_chords_and_measures_from_musicxml(musicxml_file)
    chords, counts = calculate_histogram(chord_sequence)
    normalized_counts = normalize_histogram(counts)
    entropy = calculate_entropy(normalized_counts)
    # Calculate the histogram
    
    coverage = chord_coverage(chords, measure_count)

    # 返回计算得到的参数值
    return ctnctr, pcs, mctd, ctd, entropy,coverage

import shutil
# 定义函数来计算文件夹下所有文件的参数均值
def calculate_mean_parameters(folder_path):
    # 初始化各参数的累加值
    total_ctnctr = 0
    total_pcs = 0
    total_mctd = 0
    total_ctd = 0
    total_che = 0
    total_cc = 0
    num_files = len(os.listdir(folder_path))
    # 遍历文件夹下的所有文件
    for filename in os.listdir(folder_path):
        # 构建文件的完整路径
        file_path = os.path.join(folder_path, filename)
        # 如果文件是 MusicXML 文件
        if filename.endswith('.mxl') or filename.endswith('.xml'):
            # 调用函数计算参数值
            try:
                ctnctr, pcs, mctd, ctd, che,cc = calculate_parameters(file_path)
                
                print(f"Processing {filename}...")
                # 判断ctd是否为nan
                '''if math.isnan(ctd):
                    print(f"{filename} is too long...")
                    ctd = 0
                    ctnctr = 0
                    pcs = 0
                    mctd = 0
                    che = 0
                    cc = 0
                    num_files -= 1
                '''
            
                #将该文件复制到新的文件夹
                '''if ctnctr!=0 and pcs!=0 and mctd!=0 and ctd!=0 and che!=0:
                    shutil.copy(file_path, r"E:\full_songs\test_ccd")
                    continue'''
            except:
                print(f"Error processing {filename}...")
                ctd = 0
                ctnctr = 0
                pcs = 0
                mctd = 0
                che = 0
                cc = 0
                # 跳过该文件并继续处理下一个文件
                num_files -= 1
                continue
            # ctnctr, pcs, mctd, ctd, che,cc = calculate_parameters(file_path)
            # 将参数值累加到总值上
            total_ctnctr += ctnctr
            total_pcs += pcs
            total_mctd += mctd
            total_ctd += ctd
            total_che += che
            total_cc += cc
    
    # 计算各参数的均值
    
    mean_ctnctr = total_ctnctr / num_files
    mean_pcs = total_pcs / num_files
    mean_mctd = total_mctd / num_files
    mean_ctd = total_ctd / num_files
    mean_che = total_che / num_files
    mean_cc = total_cc / num_files
    
    # 返回各参数的均值
    return mean_ctnctr, mean_pcs, mean_mctd, mean_ctd, mean_che, mean_cc
import pandas as pd
from config import *
# 调用函数计算文件夹下所有文件的参数均值

def eval_music(folder_path):
    find_all_possible_chord_progressions(folder_path,chord_folder_path)
    mean_ctnctr, mean_pcs, mean_mctd, mean_ctd, mean_che, mean_cc = calculate_mean_parameters(chord_folder_path)


    mean_ctnctr = round(mean_ctnctr, 4)
    mean_pcs = round(mean_pcs, 4)
    mean_mctd = round(mean_mctd, 4)
    mean_ctd = round(mean_ctd, 4)
    mean_che = round(mean_che, 4)
    mean_cc = round(mean_cc, 4)



    # 打印各参数的均值
    print("Mean CTnCtr:", mean_ctnctr)
    print("Mean PCs:", mean_pcs)
    print("Mean MCTD:", mean_mctd)
    print("Mean CTD:", mean_ctd)
    print("Mean Che:", mean_che)
    print("Mean CC:", mean_cc)

    return mean_ctnctr, mean_pcs, mean_mctd, mean_ctd, mean_che, mean_cc
    
    
    
    
folder_path = r"E:\论文_diff\sample"
data = []
sub_folders = [f.path for f in os.scandir(folder_path) if f.is_dir()]
for f in sub_folders:
    chord_folder_path =  f+"/chords"
    mean_ctnctr, mean_pcs, mean_mctd, mean_ctd, mean_che, mean_cc = eval_music(f)
    name = f.split("\\")[-1]
    data.append([name, mean_ctnctr, mean_pcs, mean_mctd, mean_ctd, mean_che, mean_cc])

# 定义表头
columns = ["Model", "Mean_CTNCTR", "Mean_PCS", "Mean_MCTD", "Mean_CTD", "Mean_CHE", "Mean_CC"]

# 创建一个DataFrame
df = pd.DataFrame(data, columns=columns)

# 将DataFrame写入表格（例如CSV文件）
df.to_csv("evaluation_results.csv", index=False)