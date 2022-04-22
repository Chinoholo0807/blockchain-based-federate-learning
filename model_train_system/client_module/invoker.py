import time
import web3


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

    def listen_to_event(self, event, timeout=200, poll_interval=2):
        start_time = time.time()
        current_time = time.time()
        w3_contract = web3.eth.Contract(
            address=self.contract.address,
            abi=self.contract.abi,
        )
        event_filter = w3_contract.events[event].createFilter(fromBlock="latest")
        while current_time - start_time < timeout:
            for event_resp in event_filter.get_new_entries():
                if event in event_resp:
                    return event_resp
            time.sleep(poll_interval)
            current_time = time.time()
        return None

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

    def enroll(self, dataset_desc='', extra_desc=''):
        self.contract.enrollTrain(
            dataset_desc,
            extra_desc,
            {"from": self.account}
        )

    def get_contribution(self):
        return self.contract.contributions(
            self.account,
            {"from": self.account}
        )
