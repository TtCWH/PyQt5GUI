import json

ValidationPercentage = 10
TestSetPercentage = 20
LearningRate = 0.01
LearningSteps = 5000
BatchSize = 32

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'

MODEL_DIR = 'datasets/inception_dec_2015'
MODEL_FILE = 'tensorflow_inception_graph.pb'

CACHE_DIR = '_database/bottleneck'
BOTTLENECK_PATH = '_database/bottleneck'
INPUT_DATA = 'datasets/ImageSet'

MODEL_SAVE_PATH = 'Models/'
MODEL_SAVE_NAME = 'PRModel'
Result_Save_Path = 'Results/Result.txt'
TRAININGDATABASE = '_database/'

N_CLASSES = 2
LABEL_NAME_LIST = ['非内波', '内波']

input_image_path = ''


class Parameters:
    def __init__(self, config_dictory):
        self.config = config_dictory

    def adjust_parameters(self):
        global ValidationPercentage, TestSetPercentage, LearningRate, LearningSteps, BatchSize, \
            BOTTLENECK_TENSOR_SIZE, BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME, \
            MODEL_DIR, MODEL_FILE, CACHE_DIR, INPUT_DATA, MODEL_SAVE_PATH, MODEL_SAVE_NAME, \
            Result_Save_Path, N_CLASSES, LABEL_NAME_LIST, TRAININGDATABASE, BOTTLENECK_PATH

        ValidationPercentage = self.config['ValidationPercentage']
        TestSetPercentage = self.config['TestSetPercentage']
        LearningRate = self.config['LearningRate']
        LearningSteps = self.config['LearningSteps']
        BatchSize = self.config['BatchSize']

        BOTTLENECK_TENSOR_SIZE = self.config['BOTTLENECK_TENSOR_SIZE']
        BOTTLENECK_TENSOR_NAME = self.config['BOTTLENECK_TENSOR_NAME']
        JPEG_DATA_TENSOR_NAME = self.config['JPEG_DATA_TENSOR_NAME']

        MODEL_DIR = self.config['MODEL_DIR']
        MODEL_FILE = self.config['MODEL_FILE']
        CACHE_DIR = self.config['CACHE_DIR']
        BOTTLENECK_PATH = self.config['BOTTLENECK_PATH']
        INPUT_DATA = self.config['INPUT_DATA']
        MODEL_SAVE_PATH = self.config['MODEL_SAVE_PATH']
        MODEL_SAVE_NAME = self.config['MODEL_SAVE_NAME']

        Result_Save_Path = self.config['Result_Save_Path']
        N_CLASSES = self.config['N_CLASSES']
        LABEL_NAME_LIST = self.config['LABEL_NAME_LIST']
        TRAININGDATABASE = self.config['TRAININGDATABASE']


    def save_configs(self, save_file_path):
        global ValidationPercentage, TestSetPercentage, LearningRate, LearningSteps, BatchSize, \
            BOTTLENECK_TENSOR_SIZE, BOTTLENECK_TENSOR_NAME, JPEG_DATA_TENSOR_NAME, \
            MODEL_DIR, MODEL_FILE, CACHE_DIR, INPUT_DATA, MODEL_SAVE_PATH, MODEL_SAVE_NAME, \
            Result_Save_Path, N_CLASSES, LABEL_NAME_LIST, TRAININGDATABASE

        self.config['ValidationPercentage'] = ValidationPercentage
        self.config['TestSetPercentage'] = TestSetPercentage
        self.config['LearningRate'] = LearningRate
        self.config['LearningSteps'] = LearningSteps
        self.config['BatchSize'] = BatchSize

        self.config['BOTTLENECK_TENSOR_SIZE'] = BOTTLENECK_TENSOR_SIZE
        self.config['BOTTLENECK_TENSOR_NAME'] = BOTTLENECK_TENSOR_NAME
        self.config['JPEG_DATA_TENSOR_NAME'] = JPEG_DATA_TENSOR_NAME

        self.config['MODEL_DIR'] = MODEL_DIR
        self.config['MODEL_FILE'] = MODEL_FILE
        self.config['CACHE_DIR'] = CACHE_DIR
        self.config['BOTTLENECK_PATH'] = BOTTLENECK_PATH
        self.config['INPUT_DATA'] = INPUT_DATA
        self.config['MODEL_SAVE_PATH'] = MODEL_SAVE_PATH
        self.config['MODEL_SAVE_NAME'] = MODEL_SAVE_NAME

        self.config['Result_Save_Path'] = Result_Save_Path
        self.config['N_CLASSES'] = N_CLASSES
        self.config['LABEL_NAME_LIST'] = LABEL_NAME_LIST
        self.config['TRAININGDATABASE'] = TRAININGDATABASE

        config_file = open(save_file_path, 'w')
        config_file.write(json.dumps(self.config))








