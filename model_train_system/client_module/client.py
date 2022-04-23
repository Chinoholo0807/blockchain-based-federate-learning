from client_module.invoker import Invoker
from client_module.trainer import Trainer, ModelUpdateInfo
from client_module.ipfs_client import IPFSClient, MockIPFSClient
from client_module.log import logger as l
from torch.utils.data import TensorDataset,DataLoader

import client_module.utils as u
from brownie import accounts

class Client(object):

    def __init__(self, setting):
        self.id = setting['node']["id"]
        self.tag = f"<client {self.id}>:"
        setting['node']['account'] = accounts[self.id]
        # build the invoker
        self.invoker = Invoker(setting)
        # get the contract task setting and train setting
        contract_setting = self.invoker.get_setting()
        l.debug(f'{self.tag} contract_setting:{contract_setting.train,contract_setting.task}')
        if setting.get('train') is None:
            setting['train'] = {}
        if setting.get('task') is None:
            setting['task'] = {}
        setting['train'].update(contract_setting.train)
        setting['task'].update(contract_setting.task)

        # build train dataset
        if setting['train'].get('dataset') is None:
            client_train_dataset, test_x_tensor, test_y_tensor = u.build_dataset(
                dataset_name=setting['task']['dataset_desc'],
                n_client=setting['simulate']['n_split'],
                n_attacker=setting['simulate']['n_attacker'],
                data_dir=setting['node']['dataset_dir'],
            )
            train_x_tensor, train_y_tensor = client_train_dataset[self.id]
            dataset = TensorDataset(
                train_x_tensor,
                train_y_tensor
            )
            setting['train']["dataset"] = dataset
            setting['train']['global_test_dl'] = DataLoader(
                TensorDataset(test_x_tensor, test_y_tensor),
                batch_size=5,
                shuffle=False,
            )
        dataset = setting['train']["dataset"]
        l.info(f'{self.tag} build the dataset {len(dataset)}')
        # build the trainer
        self.trainer = Trainer(setting)

        # build the ipfs client
        if setting['simulate']['mock_ipfs'] :
            l.info(f"{self.tag} use mock ipfs")
            self.ipfs = MockIPFSClient(setting)
        else:
            self.ipfs = IPFSClient(setting)

    def get_model_updates(self):
        train_infos = self.invoker.get_all_train_info()
        model_update_infos = []
        # get all update model from ipfs  with given model_hash
        for train_info in train_infos:
            bytes_param = self.ipfs.get_file(train_info.model_update_hash)
            update_info = ModelUpdateInfo(
                trainer=train_info.trainer,
                data_size=train_info.data_size,
                version=train_info.version,
                bytes_param=bytes_param,
                poll=train_info.poll
            )
            model_update_infos.append(update_info)
        l.debug(f"{self.tag} get model updates ,version {model_update_infos[-1].version}")
        return model_update_infos

    def get_global_bytes_param(self):
        model_infos = self.get_model_updates()
        global_bytes_param = self.trainer.aggregate(model_infos)
        return global_bytes_param

    def flesh_global_model(self):
        bytes_param = self.get_global_bytes_param()
        self.trainer.load_bytes_param(bytes_param)
        l.debug(f'{self.tag} flesh_global_model, view is {self.model_view()}')
        return bytes_param

    def flesh_global_model_lazy(self, bytes_param):
        self.trainer.load_bytes_param(bytes_param)
        l.debug(f'{self.tag} flesh_global_model_lazy, view is {self.model_view()}')

    def local_training(self):
        l.info(f"{self.tag} start local training...")
        self.trainer.local_training()
        l.info(f"{self.tag} finish local training...")

    def cur_model_hash(self):
        bytes_param = self.trainer.get_bytes_param()
        file_hash = self.ipfs.add_file(bytes_param)
        return file_hash

    def model_view(self):
        return self.trainer.model_view()

    def upload_train_info(self):
        data_size = self.trainer.get_data_size()
        bytes_param = self.trainer.get_bytes_param()
        file_hash = self.ipfs.add_file(bytes_param)
        e = self.invoker.upload_train_info(data_size, file_hash)
        l.info(f"{self.tag} upload train info,file hash {file_hash}")
        return e

    def evaluate(self, test_dl):
        accuracy, loss = self.trainer.evaluate(test_dl)
        return accuracy, loss

    def vote(self):
        if self.trainer.aggregate_method == 'fed_avg':
            e = self.fake_vote()
            return e
        model_infos = self.get_model_updates()
        candidates = self.trainer.get_candidates(model_infos)
        l.info(f'{self.tag} vote for {candidates}')
        e = self.invoker.vote(candidates)
        return e

    def fake_vote(self):
        candidates = []
        l.info(f'{self.tag} fake vote for {candidates}')
        e = self.invoker.vote(candidates)
        return e

    def init_model(self):
        bytes_param = self.trainer.get_bytes_param()
        init_model_hash = self.ipfs.add_file(bytes_param)
        self.invoker.init_train_info(init_model_hash)
        l.info(f"{self.tag} init model {init_model_hash}")

    def enroll(self):
        self.invoker.enroll()
        l.info(f"{self.tag} enroll")


if __name__ == "__main__":
    pass
