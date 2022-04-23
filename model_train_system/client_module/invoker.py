import time
from web3 import Web3
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
    def __init__(self, train, task):
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


EVENT_NEED_AGGREGATION = "NeedAggregation"
EVENT_NEED_VOTE = "NeedVote"
EVENT_UPLOAD = "UploadTrainInfo"


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
        return Setting(train, task)

    def get_state(self):
        version = self.contract.curVersion()
        state = self.contract.curState()
        return version, state

    def listen_to_event(self, event, timeout=5, poll_interval=1):
        l.info(f"{self.tag}wait for event {event}...")
        start_time = time.time()
        current_time = time.time()
        # TODO
        w3 = Web3(Web3.HTTPProvider("HTTP://127.0.0.1:8545"))
        assert w3.isConnected()
        w3_contract = w3.eth.contract(
            abi=self.contract.abi,
            address=self.contract.address
        )
        event_filter = w3_contract.events[event].createFilter(fromBlock="latest")
        while current_time - start_time < timeout:
            for e in event_filter.get_new_entries():
                l.debug(f"{self.tag} catch event:{e.event}({e.args.__dict__})")
                return e.args.__dict__
            time.sleep(poll_interval)
            current_time = time.time()
        l.debug(f"{self.tag} wait event {event} timeout...")
        return None

    def wait_for_aggregation(self):
        event = self.listen_to_event(EVENT_NEED_AGGREGATION)
        return event

    def wait_for_upload(self):
        event = self.listen_to_event(EVENT_UPLOAD)
        return event

    def wait_for_vote(self):
        event = self.listen_to_event(EVENT_NEED_VOTE)
        return event

    def init_train_info(self, init_model_update_hash):
        self.contract.initTrainInfo(
            init_model_update_hash,
            {"from": self.account}
        )

    def upload_train_info(self, data_size, model_update_hash):
        txn = self.contract.uploadTrainInfo(
            data_size,
            model_update_hash,
            {"from": self.account}
        )
        if EVENT_NEED_VOTE in txn.events:
            return txn.events[EVENT_NEED_VOTE]
        return None

    def get_all_train_info(self):
        train_info_list = self.contract.getTrainInfos(
            {"from": self.account}
        )

        format_train_infos = []
        for train_info in train_info_list:
            format_train_infos.append(TrainInfo(*train_info))
        return format_train_infos

    def vote(self, candidates):
        txn = self.contract.vote(
            candidates,
            {"from": self.account}
        )
        if EVENT_NEED_AGGREGATION in txn.events:
            return txn.events[EVENT_NEED_AGGREGATION]
        return None

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
