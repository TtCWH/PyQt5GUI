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

CACHE_DIR = 'datasets/bottleneck'
INPUT_DATA = 'datasets/ImageSet'

MODEL_SAVE_PATH = 'Models/'
MODEL_SAVE_NAME = 'CNNModel'
Result_Save_Path = 'Results/Result.txt'

