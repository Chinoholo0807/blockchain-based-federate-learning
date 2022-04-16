from client_module.invoker import MockInvoker
from client_module.trainer import Trainer,ModelUpdateInfo
from client_module.ipfs_client import IPFSClient,MockIPFSClient
from client_module.log import logger as l



class MockClient(object):

    def __init__(self, setting):
        self.invoker = MockInvoker(setting)
        self.trainer = Trainer(setting)

        self.ipfs = MockIPFSClient(setting)
        self.id = setting["id"]
        self.tag = f"<client {self.id}>:"

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
        return model_update_infos

    def get_global_bytes_param(self):
        model_infos = self.get_model_updates()
        global_bytes_param = self.trainer.aggregate(model_infos)
        return global_bytes_param

    def flesh_global_model(self):
        bytes_param = self.get_global_bytes_param()
        self.trainer.load_bytes_param(bytes_param)
        l.debug(f'{self.tag} flesh_global_model, view is {self.model_view()}')

    def local_train(self):
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
        self.invoker.upload_train_info(data_size, file_hash)
        l.info(f"{self.tag} upload train info,file hash {file_hash}")

    def evaluate(self, test_dl):
        accuracy, loss = self.trainer.evaluate(test_dl)
        return accuracy, loss

    def vote(self):
        model_infos = self.get_model_updates()
        candidates = self.trainer.get_candidates(model_infos)
        l.info(f'{self.tag} vote for {candidates}')
        self.invoker.vote(candidates)
    
    def init_model(self):
        bytes_param = self.trainer.get_bytes_param()
        init_model_hash = self.ipfs.add_file(bytes_param)
        self.invoker.init_train_info(init_model_hash)
        l.info(f"{self.tag} init model {init_model_hash}")

    def enroll(self):
        self.invoker.enroll()
        l.info(f"{self.tag} enroll")
        



