


class TrainInfo(object):
    def __init__(self, trainer, data_size, version, model_update_hash, poll):
        self.trainer = trainer
        self.data_size = data_size
        self.version = version
        self.model_update_hash = model_update_hash
        self.poll = poll


class MockInvoker(object):

    def __init__(self, setting):
        self.contract = setting['contract']
        self.account = setting['account']
        self.id = setting['id']

    def init_train_info(self, init_model_update_hash):
        self.contract.initTrainInfo(
            init_model_update_hash,
            {"from": self.account}
        )

    def upload_train_info(self, data_size, model_update_hash):
        self.contract.uploadTrainInfo(
            data_size,
            model_update_hash,
            {"from": self.account}
        )

    def get_all_train_info(self):
        train_info_list = self.contract.getTrainInfos(
            {"from": self.account}
        )

        format_train_infos = []
        for train_info in train_info_list:
            format_train_infos.append(TrainInfo(*train_info))
        return format_train_infos

    def vote(self, candidates):
        self.contract.vote(
            candidates,
            {"from": self.account}
        )

    def enroll(self,dataset_desc ='', extra_desc = ''):
        self.contract.enrollTrain(
            dataset_desc,
            extra_desc,
            {"from": self.account}
        )
