# Path setting
#DATASET_PATH = r"D:\learn_project\deepchoir-main_第一版代码 - 副本\dataset"
#DATASET_PATH = r"E:\论文_中文\绘图"

DATASET_PATH = r"E:\dataset_清洗_splited"

CORPUS_PATH = "ccd_chord(root_intervals_duration)_melodytoken129_320.bin"
# CORPUS_PATH = "ccd_melodytoken129_320.bin"
#CORPUS_PATH = "D:\learn_project\deepchoir-main_第一版代码 - 副本\BS_unenhanced_withchord.bin"
WEIGHTS_PATH = 'My_model_notot_test.hdf5'
#WEIGHTS_PATH = 'My_model_withchord_diff_withoutpretrain_epoch200_batch256_num_layer3_rnn128_nopadding_overtrained_melodytoken129.hdf5'
#INPUTS_PATH = r"D:\learn_project\deepchoir-main\inputs"
INPUTS_PATH = r"E:\论文_diff\sample1"
#INPUTS_PATH = r"E:\dataset_清洗_input"
#INPUTS_PATH = r"E:\完整歌曲_input"
OUTPUTS_PATH = r"E:\论文_diff\sample_ori_model_latent_4_24"
#OUTPUTS_PATH = "E:\music_dataset\output"
# 'loader.py'
EXTENSION = ['.musicxml', '.xml', '.mxl']

# 'model.py'
VAL_RATIO = 0.3
DROPOUT = 0
# CONDITION=['chord', 'condition',"melody"]
CONDITION=['chord']
SEGMENT_LENGTH =320
RNN_SIZE =128
NUM_LAYERS = 3
BATCH_SIZE = 32
EPOCHS = 50
NUM_HEADS = 4
MERGE_LAYER = 3
CODELENGTH = 32
VAE_PATH = 'autodencoder_dense_crosssig.hdf5'
# 'choralizer.py'
HARMONICITY = 0.8
HOMOPHONICITY = 0.5
PRE_NOISE = False
KEEP_CHORD = True
VAL =True
WATER_MARK = False
SEED = 8  