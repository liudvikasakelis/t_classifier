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
                   [256, 2, None],
                   [256, 2, None],
                   [256, 2, None],
                   [256, 2, 2]]
    
    fully_connected_layers = [1024, 1024]
    th = 1e-6

    embedding_size = 128
    
    
class Config(object):
    alphabet = "abcdefghijklmnopqrstuvwxyząčęėįšųūž0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}"
    alphabet_size = len(alphabet)
    l0 = 100
    batch_size = 128
    num_of_classes = 2
    dropout_p = 0.5
    train_data_source = 'data/ag_news_csv/cut_train.csv'
    dev_data_source = 'data/ag_news_csv/cut_test.csv'
    
    training = TrainingConfig()
    
    model = ModelConfig()

config = Config()














