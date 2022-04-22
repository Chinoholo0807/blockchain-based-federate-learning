import os

train_setting = {
    "batch_size": 10,
    "learning_rate": "0.01",
    "epochs": 2,
    "n_trainer": 10,
    "n_vote": 6,
}
task_setting = {
    "task_description": "Federate Learning Computing Market Task",
    "model_description": "mnist_2nn",
    # "model_description":"emnist_cnn",
    "dataset_description": "MNIST",
    # "dataset_description":"EMNIST",
    "max_version": 1000,
    "n_client": 10,
}

setting = {}
for k in train_setting.keys():
    setting[k] = train_setting[k]
for k in task_setting.keys():
    setting[k] = task_setting[k]

setting['learning_rate'] = float(train_setting['learning_rate'])
setting['report_dir'] = \
    r'.\reports'
setting['dataset_dir'] = \
    r'.\data'
setting['log_dir'] = os.path.join(setting['report_dir'], 'logs')
setting['results_dir'] = os.path.join(setting['report_dir'], 'results')
setting['model_dir'] = os.path.join(setting['report_dir'],'model')
setting['n_split'] = 50
setting['n_attacker'] = 3
setting['aggregate_method'] = 'fed_vote_avg'

setting['ipfs_api'] = "/ip4/127.0.0.1/tcp/5001"