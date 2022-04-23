import hashlib
import pickle
import numpy as np
import torch
import copy
import math
from torch.utils.data import DataLoader, TensorDataset
import client_module.model as model
from client_module.log import logger as l


class ModelUpdateInfo(object):

    def __init__(self, trainer, data_size, version, bytes_param, poll=1, bytes_model_hash=None):
        self.trainer = trainer
        self.data_size = data_size
        self.version = version
        self.bytes_model = model
        self.poll = poll
        self.bytes_model_hash = bytes_model_hash
        self.param_dict = pickle.loads(bytes_param)


class Trainer(object):

    def __init__(self, setting):
        self.id = setting['node']['id']
        self.tag = f"<trainer {self.id}>:"

        train = setting['train']
        task = setting['task']
        # training setting
        self.epochs = train['epochs']
        self.batch_size = train['batch_size']
        self.learning_rate = train['learning_rate']
        self.n_poll = train['n_poll']
        self.n_trainer = train['n_trainer']
        self.train_ds = train['dataset']
        self.aggregate_method = train['aggregate_method']

        self.model_name = task['model_desc']


        assert isinstance(self.train_ds, TensorDataset)


        # init training model
        self.dev = torch.device('cpu')
        if torch.cuda.is_available():
            l.info(f'{self.tag} use cuda as dev')
            self.dev = torch.device('cuda')
        if torch.cuda.device_count() > 1:
            self.model = torch.nn.DataParallel(self.model)

        self.model = model.get_model(self.model_name)
        self.model = self.model.to(self.dev)

        self.opti = model.get_opti(self.model, self.learning_rate)

        self.loss_fn = model.get_loss_fn()

        l.debug(f"{self.tag} opti({id(self.opti)}) model({id(self.model)}) loss({id(self.loss_fn)})")

        # init data loader
        self.train_dl = DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True)
        self.test_dl = self.train_dl
        if not (train.get('global_test_dl') is None):
            self.global_test_dl = train['global_test_dl']
    def load_bytes_param(self, bytes_param):
        assert isinstance(bytes_param, bytes)
        param_dict = pickle.loads(bytes_param)
        self._load_model(param_dict)

    def get_bytes_param(self):
        model_param_dict = self.model.state_dict()
        # transfer obj to bytes
        bytes_param = pickle.dumps(model_param_dict)
        return bytes_param

    def get_param_dict(self):
        return self.model.state_dict()

    def load_param_dict(self,param_dict):
        self._load_model(param_dict)

    def get_data_size(self):
        return len(self.train_dl)

    def get_model_abstract(self):
        m = hashlib.md5()
        m.update(self.get_bytes_param())
        return m.hexdigest()

    def model_view(self):
        sd = self.model.state_dict()
        return torch.flatten(sd[list(sd.keys())[0]])[:4]

    def _load_model(self, param_dict):
        d = copy.deepcopy(param_dict)
        self.model.load_state_dict(d, strict=True)

    def local_training(self):
        for epoch in range(self.epochs):
            for train_x, train_y in self.train_dl:
                train_x, train_y = train_x.to(self.dev), train_y.to(self.dev).long()
                self.opti.zero_grad()
                pred_y = self.model(train_x)
                loss = self.loss_fn(pred_y, train_y)
                loss.backward()
                self.opti.step()

    def evaluate_global(self):
        return self.evaluate(self.global_test_dl)


    def evaluate(self, test_dl):
        """
        return the accuracy and loss
        """
        correct = 0
        data_size = 0
        running_loss = 0.0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for data in test_dl:
                test_x, test_y = data
                test_x = test_x.to(self.dev)
                test_y = test_y.to(self.dev).long()
                # calculate outputs by running test_x through the nn
                outputs = self.model(test_x)
                loss = self.loss_fn(outputs, test_y)
                running_loss += loss
                _, predicted = torch.max(outputs.data, 1)
                data_size += test_y.size(0)
                correct += (predicted == test_y).sum().item()
            # running_loss = running_loss.item() / data_size * test_dl.batch_size
            running_loss = running_loss.item() / data_size
            acc = correct / data_size
            return acc, running_loss

    def get_candidates(self, model_update_infos):
        """
        get the candidate list for vote
        """
        accs = []
        for m in model_update_infos:
            self.model.load_state_dict(m.param_dict, strict=True)
            acc, _ = self.evaluate(self.test_dl)
            accs.append(acc)

        args = np.argsort(accs)[::-1][:self.n_poll]
        l.debug(f'{self.tag} accs evaluated is {accs},choice idx is {args}')
        candidates = []
        for idx in args:
            candidates.append(model_update_infos[idx].trainer)
        return candidates

    # return the bytes_param after aggregation
    def aggregate(self, model_infos) -> bytes:
        if len(model_infos) == 0:
            return pickle.dumps(self.model.state_dict())
        aggregate_param = None
        if len(model_infos) == 1:
            aggregate_param = model_infos[0].param_dict
        elif self.aggregate_method == 'fed_vote_avg':
            l.info("use fed_vote_avg to aggregate")
            aggregate_param = self.aggregate_fed_vote_avg(model_infos)
        else:
            l.info("use fed_avg to aggregate")
            aggregate_param = self.aggregate_fed_avg(model_infos)
        self.model.load_state_dict(aggregate_param, strict=True)
        bytes_param = pickle.dumps(aggregate_param)
        return bytes_param

    def aggregate_fed_avg(self, model_infos) -> dict:
        average_params = None
        all_data_size = 0
        for model_info in model_infos:
            all_data_size = all_data_size + model_info.data_size
        log_infos = ''
        for model_info in model_infos:
            fraction = model_info.data_size / all_data_size
            log_infos = log_infos + f" trainer{model_info.trainer}_poll{model_info.poll}_f{fraction}"
            if average_params is None:
                average_params = {}
                for k, v in model_info.param_dict.items():
                    average_params[k] = v.clone() * fraction
            else:
                for k in average_params:
                    average_params[k] = average_params[k] + model_info.param_dict[k] * fraction
        l.debug(self.tag+log_infos)
        return average_params

    def aggregate_fed_vote_avg(self, model_infos) -> dict:
        average_params = None
        denominator = 0

        def sigmoid(z):
            return 1 / (1 + math.exp(-z))

        for model_info in model_infos:
            denominator += model_info.data_size * sigmoid(model_info.poll - self.n_poll + 1)
        log_infos = ''
        for model_info in model_infos:
            fraction = model_info.data_size * sigmoid(model_info.poll - self.n_poll + 1) / denominator
            log_infos = log_infos + f" trainer{model_info.trainer}_poll{model_info.poll}_f{fraction}"
            if average_params is None:
                average_params = {}
                for k, v in model_info.param_dict.items():
                    average_params[k] = v.clone() * fraction
            else:
                for k in average_params:
                    average_params[k] = average_params[k] + model_info.param_dict[k] * fraction
        l.debug(self.tag+log_infos)
        return average_params
