config = {} 
class TrainingConfig(object):
    
    alpha = 2e-3
    decay = 0
    beta1 = 0.9
    beta2 = 0.999
    epochs = 15
    evaluate_every = 1
    checkpoint_every = 1
    epsilon = 1e-8

class ModelConfig(object):
    conv_layers = [[256, 7, 3],
                   [256, 7, 3],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, 3]]
    
    fully_connected_layers = [1024, 1024]
    th = 1e-6
    embedding_size = 128
    
        
class Config(object):
    # alphabet = "abcdefghijklmnopqrst.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    alphabet_size = 400
    l0 = 125
    batch_size = 512
    num_of_classes = 112
    dropout_p = 0.5
    date_cutoff = 16100
    # train_data_source = 'data/words/train.csv'
    # dev_data_source = 'data/words/test.csv'
    
    training = TrainingConfig()
    
    model = ModelConfig()

config = Config()

