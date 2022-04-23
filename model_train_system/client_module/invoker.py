import time
import web3
from brownie import ModelTrain, Contract
from client_module.log import logger as l

class TrainInfo(object):
    def __init__(self, trainer, data_size, version, model_update_hash, poll):
        self.trainer = trainer
        self.data_size = data_size
        self.version = version
        self.model_update_hash = model_update_hash
        self.poll = poll

class Setting(object):
    def __init__(self,train,task):
        self.train = {}
        self.train['batch_size'] = train[0]
        self.train['learning_rate'] = train[1]
        if type(self.train['learning_rate']) == str:
            self.train['learning_rate'] = float(self.train['learning_rate'])
        self.train['epochs'] = train[2]
        self.train['n_trainer'] = train[3]
        self.train['n_poll'] = train[4]
        self.train['n_client'] = train[5]
        self.train['max_version'] = train[6]

        self.task = {}
        self.task['task_desc'] = task[0]
        self.task['model_desc'] = task[1]
        self.task['dataset_desc'] = task[2]

class Invoker(object):
    def __init__(self, setting):
        self.id = setting['node']['id']
        self.tag = f"<invoker {self.id}>:"
        self.account = setting['node']['account']
        self.contract = Contract.from_abi(
            "ModelTrain",
            setting['node']['contract_addr'],
            ModelTrain.abi
        )
        l.info(f"{self.tag}interface with contract {self.contract}")

    def get_setting(self):
        [task, train] = self.contract.setting()
        return Setting(train,task)

    def listen_to_event(self, event, timeout=200, poll_interval=2):
        l.info(f"{self.tag}wait for event {event}...")
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

    def wait_for_aggregate(self):
        resp = self.listen_to_event('NeedAggregation', 2000000)
        return resp

    def wait_for_vote(self):
        resp = self.listen_to_event('NeedVote',2000000)
        return resp

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
