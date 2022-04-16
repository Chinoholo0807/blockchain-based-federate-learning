import os

train_setting = {
    "batch_size": 10,
    "learning_rate": "0.01",
    "epochs": 2,
    "n_trainer": 5,
    "n_vote": 3,
}
task_setting = {
    "task_description": "Federate Learning Computing Market Task",
    "model_description": "mnist_2nn",
    # "model_description":"emnist_2nn",
    "dataset_description": "MNIST",
    # "dataset_description":"EMNIST",
}

setting = {}
for k in train_setting.keys():
    setting[k] = train_setting[k]
for k in task_setting.keys():
    setting[k] = task_setting[k]

setting['report_dir'] = \
    r'.\reports'
setting['dataset_dir'] = \
    r'.\data'
setting['log_dir'] = os.path.join(setting['report_dir'], 'logs')
setting['results_dir'] = os.path.join(setting['report_dir'], 'results')

setting['n_attacker'] = 4
setting['aggregate_method'] = 'fed_vote_avg'
