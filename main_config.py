# Settings
seed = [42]
model = ["UniXcoder"]
train_dataset = ["csn"] 

data_type = ["hard_ambiguous"]
use_data_augmentation = ["True"]

# Hyperparameter
learning_rate = [2e-5]
train_batch_size = [32] 
eval_batch_size = [64] 
epoch = [10] 

tuning_param  = ["seed", "model", "train_dataset", "learning_rate", "train_batch_size", "eval_batch_size", "epoch", "data_type", "use_data_augmentation"] 
param = {"seed": seed, "model": model, "train_dataset": train_dataset, "learning_rate": learning_rate, "train_batch_size": train_batch_size, "eval_batch_size": eval_batch_size, "epoch": epoch, "data_type": data_type, "use_data_augmentation": use_data_augmentation}
