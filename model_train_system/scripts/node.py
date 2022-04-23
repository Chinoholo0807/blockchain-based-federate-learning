from client_module.log import logger as l
from client_module.log import init_logging
from client_module.client import Client
import scripts.helpful_scripts as script
from brownie import accounts
import time


class Node(Client):

    def __init__(self, setting_path):
        setting = script.read_yaml(setting_path)
        init_logging(log_level_str=setting['node']['log_level'])

        self.id = setting['node']['id']
        self.is_uploader = setting['node']['is_uploader']
        self.tag = f'<node {self.id}>:'
        setting['node']['account'] = accounts[self.id]
        super().__init__(setting)
        l.info(f'{self.tag} init success')

    # when stop,return true
    def loop_one_time(self):
        self.flesh_global_model()
        acc, loss = self.trainer.evaluate_global()
        l.info(f"{self.tag} evaluate global model with global test dataset,acc:{acc},loss:{loss}")
        cur_version, cur_state = self.invoker.get_state()
        l.info(f"{self.tag} cur_version:{cur_version} cur_state:{cur_state}")
        if cur_state == 2:
            return True
        if cur_state != 0:
            # sleep and retry
            time.sleep(2)
            return False


        self.local_training()
        acc, loss = self.trainer.evaluate_global()
        l.info(f"{self.tag} evaluate local model with global test dataset,acc:{acc},loss:{loss}")
        ########################################
        e = self.upload_train_info()
        if e:
            l.info(f"{self.tag} uploadTrainInfo txn's event NeedVote {e}")
        else:
            while True:
                e = self.invoker.wait_for_vote()
                if e is None:
                    latest_version, latest_state = self.invoker.get_state()
                    if latest_version != cur_version:
                        l.info(
                            f"{self.tag} cur_version {cur_version} do not match latest_version {latest_version} ,skip...")
                        return False
                    # contract is wait for vote , event is expired, can not catch it ~
                    if latest_state == 1:
                        l.info(f"{self.tag} latest state is need vote,now can vote ,continue...")
                        break
                else:
                    l.info(f"{self.tag} catch event NeedVote {e}")
                    break
        ########################################
        e = self.vote()
        if e:
            l.info(f"{self.tag} vote txn's NeedAggregation {e}")
        else:
            while True:
                e = self.invoker.wait_for_aggregation()
                if e is None:
                    latest_version, latest_state = self.invoker.get_state()
                    if latest_version > cur_version:
                        l.info(
                            f"{self.tag} latest version {latest_version} > cur version {cur_version},now can aggregate the latest model,continue...")
                        break
                else:
                    l.info(f"{self.tag} catch event NeedAggregation {e}")
                    break
        return False

    def run(self):
        self.enroll()
        if self.is_uploader:
            self.init_model()
        stop = False
        while not stop:
            stop = self.loop_one_time()

        l.info(f"{self.tag} run finish")

def start(setting_path):
    # args = parser.parse_args()
    # setting_path = args.setting_path
    print('create node with setting_path:', setting_path)
    setting = script.read_yaml(setting_path)
