# inferenceoptions.py
class InferenceOptions:
    def __init__(self):
        self.dataset_mode = 'test'  # change this if your dataset mode is different
        self.num_threads = 0  # test code only supports num_threads = 0
        self.batch_size = 1  # test code only supports batch_size = 1
        self.serial_batches = True  # disable data shuffling
        self.no_flip = True  # no flip
        self.display_id = -1  # no visdom display
        self.model = 'test'  # change this to your model type
        self.epoch = 'latest'  # use the latest model
        self.load_iter = 0  # load_iter is 0 by default
        self.verbose = False  # if verbose, print detailed information about the model
        self.results_dir = './results/'  # results will be saved here
        self.aspect_ratio = 1.0  # aspect ratio of result images
        self.phase = 'test'  # phase should be set to 'test'
        self.eval = True  # use eval mode during test time
        self.num_test = float('inf')  # number of test images
        self.isTrain = False  # this is test code