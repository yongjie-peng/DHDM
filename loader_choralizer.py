from copy import deepcopy
import os
import pickle
import numpy as np

from music21 import *
from config import *
from tqdm import trange


def ks2gap(ks):
    if isinstance(ks, key.KeySignature):
        ks = ks.asKey()

    try:
        # Identify the tonic
        if ks.mode == 'major':
            tonic = ks.tonic

        else:
            tonic = ks.parallel.tonic

    except:
        return interval.Interval(0)

    # Transpose score
    gap = interval.Interval(tonic, pitch.Pitch('C'))

    return gap.semitones


def split_by_key(score):
    scores = []
    score_part = []
    ks_list = []
    ks = None
    ts = meter.TimeSignature('c')
    pre_offset = 0

    for element in score.flat:

        # If is key signature
        if isinstance(element, key.KeySignature) or isinstance(element, key.Key):

            # If is not the first key signature
            if ks != None:

                scores.append(stream.Stream(score_part))
                ks = element
                ks_list.append(ks)
                pre_offset = ks.offset
                ks.offset = 0
                new_ts = meter.TimeSignature(ts.ratioString)
                score_part = [ks, new_ts]

            else:

                ks = element
                ks_list.append(ks)
                score_part.append(ks)

        # If is time signature
        elif isinstance(element, meter.TimeSignature):

            element.offset -= pre_offset
            ts = element
            score_part.append(element)

        else:

            element.offset -= pre_offset
            score_part.append(element)

    scores.append(stream.Stream(score_part))
    if ks_list == []:
        ks_list = [key.KeySignature(0)]

    gap_list = [ks2gap(ks) for ks in ks_list]

    return scores, gap_list


def quant_score(score):
    for element in score.flat:
        onset = np.ceil(element.offset / 0.25) * 0.25

        if isinstance(element, note.Note) or isinstance(element, note.Rest) or isinstance(element, chord.Chord):
            offset = np.ceil((element.offset + element.quarterLength) / 0.25) * 0.25
            element.quarterLength = offset - onset

        element.offset = onset

    return score


def get_filenames(input_dir):
    filenames = []

    # Traverse the path
    for dirpath, dirlist, filelist in os.walk(input_dir):
        # Traverse the list of files
        for this_file in filelist:
            # Ensure that suffixes in the training set are valid
            if input_dir == DATASET_PATH and os.path.splitext(this_file)[-1] not in EXTENSION:
                continue
            filename = os.path.join(dirpath, this_file)
            filenames.append(filename)

    return filenames


def beat_seq(ts):
    # Read time signature
    beatCount = ts.numerator
    beatDuration = 4 / ts.denominator

    # Create beat sequence
    beat_sequence = [0] * beatCount * int(beatDuration / 0.25)
    beat_sequence[0] += 1

    # Check if the numerator is divisible by 3 or 2
    medium = 0

    if (ts.numerator % 3) == 0:
        medium = 3

    elif (ts.numerator % 2) == 0:
        medium = 2

    for idx in range(len(beat_sequence)):

        # Add 1 to each beat
        if idx % ((beatDuration / 0.25)) == 0:
            beat_sequence[idx] += 1

        # Mark medium-weight beat (at every second or third beat)
        if (medium == 3 and idx % ((3 * beatDuration / 0.25)) == 0) or \
                (medium == 2 and idx % ((2 * beatDuration / 0.25)) == 0):
            beat_sequence[idx] += 1

    return beat_sequence


def melody_reader(melody_part, gap):
    # Initialization
    melody_txt = []
    ts_seq = []
    beat_txt = []
    fermata_txt = []
    chord_txt = []
    chord_token = [0.] * 12
    fermata_flag = False

    # Read note and meta information from melody part
    for element in melody_part.flat:

        if isinstance(element, note.Note):
            # midi pitch as note onset
            token = element.transpose(gap).pitch.midi

            for f in element.expressions:
                if isinstance(f, expressions.Fermata):
                    fermata_flag = True
                    break

        elif isinstance(element, note.Rest):
            # 128 as rest onset
            token = 128

        elif isinstance(element, chord.Chord) and not isinstance(element, harmony.ChordSymbol):
            notes = [n.transpose(gap).pitch.midi for n in element.notes]
            notes.sort()
            token = notes[-1]

        elif isinstance(element, harmony.ChordSymbol):
            element = element.transpose(gap)
            chord_token = [0.] * 12
            for n in element.pitches:
                chord_token[n.midi % 12] += 1.
            continue

        # Read the current time signature
        elif isinstance(element, meter.TimeSignature):

            ts_seq.append(element)
            continue

        else:
            continue

        if element.quarterLength == 0:
            continue

        melody_txt += [token] + [129] * (int(element.quarterLength * 4) - 1)
        
        #melody_txt += [token] + [token] * (int(element.quarterLength * 4) - 1)
        fermata_txt += [int(fermata_flag)] * int(element.quarterLength * 4)
        chord_txt += [chord_token] * int(element.quarterLength * 4)
        fermata_flag = False

    # Initialization
    cur_cnt = 0
    pre_cnt = 0
    beat_sequence = beat_seq(meter.TimeSignature('c'))

    # create beat sequence
    if len(ts_seq) != 0:

        # Traverse time signartue sequence
        for ts in ts_seq:

            # Calculate current time step
            cur_cnt = ts.offset / 0.25

            if cur_cnt != 0:

                # Fill in the previous beat sequence
                beat_txt += beat_sequence * int((cur_cnt - pre_cnt) / len(beat_sequence))

                # Complete the beat sequence
                missed_beat = int((cur_cnt - pre_cnt) % len(beat_sequence))

                if missed_beat != 0:
                    beat_txt += beat_sequence[:missed_beat]

            # Update variables
            beat_sequence = beat_seq(ts)
            pre_cnt = cur_cnt

    # Handle the last time signature
    cur_cnt = len(melody_txt)
    beat_txt += beat_sequence * int((cur_cnt - pre_cnt) / len(beat_sequence))

    # Complete the beat sequence
    missed_beat = int((cur_cnt - pre_cnt) % len(beat_sequence))

    if missed_beat != 0:
        beat_txt += beat_sequence[:missed_beat]

    if len(melody_txt) != len(beat_txt) or len(beat_txt) != len(fermata_txt) or len(fermata_txt) != len(chord_txt):
        print('Warning')

    return melody_txt, beat_txt, fermata_txt, chord_txt



def pad_list_with_zeros(lst, max_length):
    # 计算需要插入的0的数量
    padding_length = max_length - len(lst)
    # 插入相应数量的0
    lst.extend([0] * padding_length)


def make_lists_equal_length(lists):
    # 找到最大的子列表长度
    max_length = max(len(lst) for lst in lists)

    # 对每个列表进行插值0，使其长度与最大长度相同
    for lst in lists:
        pad_list_with_zeros(lst, max_length)

def mid_to_chords(mid_file):
    # 读取mid文件，创建一个music21对象
    score = converter.parse(mid_file)
    # 获取所有的和弦
    for ids,part in enumerate(score):
        chords = score.flat.getElementsByClass(chord.Chord)
        # 初始化输出数组
        output_root = []
        output_intervals = []
        output_duration = []
        # 遍历所有的和弦
        for c in chords:
            # 获取和弦的根音
            root = c.root()
            # 获取和弦的音程集合
            intervals = c.normalOrder
            root_num = root.midi
            # 根据根音和音程集合，判断和弦的类别，用一个整数表示，从0到23，分别表示24种不同的和弦类型
            intervals_one_hot = []
            for i in range(12):
                if i in intervals:
                    intervals_one_hot.append(1)
                else:
                    intervals_one_hot.append(0)

            if len(intervals)==0:
                print(mid_file)
            # 获取和弦的持续时间（以四分音符为单位）
            duration = int(float( c.duration.quarterLength)*4)
            # 把和弦类别和持续时间作为一个列表添加到输出数组中
            output_root.append(root_num)
            output_duration.append(duration)
            output_intervals.append(intervals_one_hot)
    # 返回输出数组
    return output_root,output_intervals,output_duration

def convert_files(filenames, fromDataset=True):
    print('\nConverting %d files...' % (len(filenames)))
    soprano_melody = []
    soprano_beat = []
    soprano_fermata = []
    soprano_chord = []
    data_corpus = []
    pad_value = 0
    
    alto = []
    tenor = []
    bass = []
    chord_root = []
    chord_intervals = []
    chord_duration = []
    for filename_idx in trange(len(filenames)):

        # Read this music file
        filename = filenames[filename_idx]
        if not fromDataset:

            chord_root = []
            chord_duration = []
            chord_intervals = []

        # Ensure that suffixes are valid
        if os.path.splitext(filename)[-1] not in EXTENSION:
            continue


        root, intervals, duration = mid_to_chords(filename)
        

        # try:
        # Read this music file
        try:
            score = converter.parse(filename)

        except:
            print(filename)
        # Read each part
        if not fromDataset:
                
                soprano_melody = []
                soprano_beat = []
                soprano_fermata = []
                alto = []
                tenor = []
                bass = []
                soprano_chord = []
        for idx, part in enumerate(score.parts):
                

            part = quant_score(part)
            splited_score, gap_list = split_by_key(part)

            if idx == 0:
                original_score = deepcopy(part)
                # Convert soprano
                for s_idx in range(len(splited_score)):
                    melody_part = splited_score[s_idx]
                    melody_txt, beat_txt, fermata_txt, chord_txt = melody_reader(melody_part, gap_list[s_idx])

                    soprano_melody.append(melody_txt)
                    soprano_beat.append(beat_txt)
                    soprano_fermata.append(fermata_txt)
                    soprano_chord.append(chord_txt)


                '''if not fromDataset:
                    break'''

            else:

                # Convert alto, tenor and bass
                for s_idx in range(len(splited_score)):
                    melody_part = splited_score[s_idx]
                    melody_txt, beat_txt, fermata_txt, chord_txt = melody_reader(melody_part, gap_list[s_idx])

                if idx == 1:
                    alto.append(melody_txt)

                elif idx == 2:
                    tenor.append(melody_txt)

                elif idx == 3:
                    bass.append(melody_txt)
        '''if len(alto) != len(tenor) or len(bass) != len(tenor) or len(soprano_melody) != len(tenor) :
            l = min( len(alto),len(tenor),len(bass),len(soprano_melody))
            alto = alto[:l]
            tenor = tenor[:l]
            bass = bass[:l]
            soprano_melody = soprano_melody[:l]
            print(filename)'''
        pad_list = [0 for _ in range(12)]
        pad_song_chord = [0 for _ in range(len(soprano_melody[-1]))]
        if not root:
            
            root = pad_song_chord
            intervals = [pad_list for x in range(len(soprano_melody[-1]))]
            duration = pad_song_chord

        chord_root.append(root)
        chord_intervals.append(intervals)
        chord_duration.append(duration)
        try:

            '''max_len2 = max(len(alto[-1]), len(bass[-1]), len(tenor[-1]), len(soprano_melody[-1]), len(soprano_beat[-1]),
                           len(soprano_fermata[-1]), len(soprano_chord[-1]),len(chord_root),len(chord_intervals),len(chord_duration),300)

            alto[-1] = alto[-1]+[pad_value] * (max_len2 - len(alto[-1]))

            tenor[-1] = tenor[-1]+[pad_value] * (max_len2 - len(tenor[-1]))
            bass[-1] = bass[-1]+[pad_value] * (max_len2 - len(bass[-1]))

            soprano_melody[-1] = soprano_melody[-1]+[pad_value] * (max_len2 - len(soprano_melody[-1]))
            soprano_fermata[-1] = soprano_fermata[-1]+[pad_value] * (max_len2 - len(soprano_fermata[-1]))
            soprano_beat[-1] = soprano_beat[-1]+[pad_value] * (max_len2 - len(soprano_beat[-1]))

            chord_root[-1] = chord_root[-1]+[pad_value] * (max_len2 - len(chord_root[-1]))
            chord_intervals[-1] = chord_intervals[-1]+[pad_list] * (max_len2 - len(chord_intervals[-1]))
            chord_duration[-1] = chord_duration[-1]+[pad_value] * (max_len2 - len(chord_duration[-1]))'''
            #max_len2 = min(300, max(len(alto[-1]), len(bass[-1]), len(tenor[-1]), len(soprano_melody[-1]), len(soprano_beat[-1]),len(soprano_fermata[-1]), len(soprano_chord[-1]),len(chord_root),len(chord_intervals),len(chord_duration)))
            max_len2 = SEGMENT_LENGTH
            alto[-1] = alto[-1][:max_len2] + [pad_value] * (max_len2 - len(alto[-1]))
            tenor[-1] = tenor[-1][:max_len2] + [pad_value] * (max_len2 - len(tenor[-1]))
            bass[-1] = bass[-1][:max_len2] + [pad_value] * (max_len2 - len(bass[-1]))
            soprano_melody[-1] = soprano_melody[-1][:max_len2] + [pad_value] * (max_len2 - len(soprano_melody[-1]))
            soprano_fermata[-1] = soprano_fermata[-1][:max_len2] + [pad_value] * (max_len2 - len(soprano_fermata[-1]))
            soprano_beat[-1] = soprano_beat[-1][:max_len2] + [pad_value] * (max_len2 - len(soprano_beat[-1]))
            chord_root[-1] = chord_root[-1][:max_len2] + [pad_value] * (max_len2 - len(chord_root[-1]))
            chord_intervals[-1] = chord_intervals[-1][:max_len2] + [pad_list] * (max_len2 - len(chord_intervals[-1]))
            chord_duration[-1] = chord_duration[-1][:max_len2] + [pad_value] * (max_len2 - len(chord_duration[-1]))

        except:
            max_len2 = 0

        
        if not fromDataset:
            data_corpus.append(
                [soprano_melody, soprano_beat, soprano_fermata,  chord_root,chord_intervals,chord_duration, original_score, gap_list, filename, alto, tenor, bass])

    if fromDataset:
        data_corpus = [[soprano_melody, soprano_beat, soprano_fermata, chord_root,chord_intervals,chord_duration], alto, tenor, bass, filenames]

        with open(CORPUS_PATH, "wb") as filepath:
            pickle.dump(data_corpus, filepath)

    else:
        return data_corpus


if __name__ == "__main__":
    # Read encoded music information and file names
    filenames = get_filenames(input_dir=DATASET_PATH)
    convert_files(filenames)