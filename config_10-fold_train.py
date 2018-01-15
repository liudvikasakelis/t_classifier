config = {} 
class TrainingConfig(object):
    
    base_rate = 1e-2
    momentum = 0.9
    decay_step = 15000
    decay_rate = 0.95
    epochs = 1
    evaluate_every = 100
    checkpoint_every = 100

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
    l0 = 450
    batch_size = 512
    num_of_classes = 112
    dropout_p = 0.5
    # train_data_source = 'data/words/train.csv'
    # dev_data_source = 'data/words/test.csv'
    
    training = TrainingConfig()
    
    model = ModelConfig()

config = Config()

