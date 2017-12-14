from datetime import *

class Config:
    """Holds model hyperparams and data information.

    The config class is used to store various hyperparameters and dataset
    information parameters. Model objects are passed a Config() object at
    instantiation.
    """
    vocab_size = 50000
    vocab_size1 = None
    vocab_size2 = None
    dropout = 0.667
    hidden_size = 256
    mini_batch = 20
    batch_size = 64
    n_epochs = 2
    lr = 0.001
    lamda = 4.
    beta = 1.

    def __init__(self, args):
        self.lang1 = args.lang1
        self.lang2 = args.lang2
        self.embed_size = args.dim

        if args.model_path is not None:
            # Where to save things.
            self.output_path = args.model_path
        else:
            self.output_path = "results/{}/{:%Y%m%d_%H%M%S}/".format(self.lang1 + '-' + self.lang2 + '.' + self.embed_size + 'd', datetime.now())
        self.model_output = self.output_path + "model.weights"
        self.eval_output = self.output_path + "results.txt"
        self.log_output = self.output_path + "log"