import os

import pandas as pd
import numpy as np
import time
from brownie import accounts
from client_module.log import logger as l, init_logging
from scripts.deploy import deploy_contract_setting
import client_module.utils as u
from client_module.client import Client
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from scripts.helpful_scripts import read_yaml
import torch


class ClusterSimulator(object):

    def __init__(self,setting_path):
        setting = read_yaml(setting_path)
        self.setting = setting
        init_logging(setting['node']['log_dir'], setting['node']['log_level'])
        l.info('\ntrain:{train}\ntask:{task}\nsimulate:{simulate}\n'.format(
            train = setting['train'],
            task = setting['task'],
            simulate = setting['simulate'],
        ))
        self.n_attacker = setting['simulate']['n_attacker']
        self.accounts = accounts

        self.n_client = setting['train']['n_client']
        self.n_trainer = setting['train']['n_trainer']
        self.n_poll = setting['train']['n_poll']

        self.model_name = setting['task']['model_desc']

        # build and split dataset
        client_train_dataset, test_x_tensor, test_y_tensor = u.build_dataset(
            dataset_name=setting['task']['dataset_desc'],
            n_client=setting['simulate']['n_split'],
            n_attacker=self.n_attacker,
            data_dir=setting['node']['dataset_dir']
        )
        self.test_dl = DataLoader(
            TensorDataset(test_x_tensor, test_y_tensor),
            batch_size=5,
            shuffle=False
        )
        l.info(f"build test dataloader finish...")
        self.contract = deploy_contract_setting(setting)
        l.info(f"deploy contract,contract's info:{self.contract}")

        self.round = 0
        self.train_result = []
        self.contributions_result = []

        self.clients = []
        for id in range(self.n_client):
            client_setting = setting.copy()
            train_x_tensor, train_y_tensor = client_train_dataset[id]
            client_setting['train']["dataset"] = TensorDataset(
                train_x_tensor,
                train_y_tensor
            )
            client_setting['node']['id'] = id
            client_setting['node']["account"] = self.accounts[id]
            client_setting['node']["contract_addr"] = self.contract.address

            new_client = Client(client_setting)
            self.clients.append(new_client)

        l.info(f"build {self.n_client} clients success")

    def flesh(self, client_ids=None):
        """
        flesh the global aggregated model into local model
        """
        l.info('[flesh global model]')
        selected_clients = []
        if client_ids is None:
            selected_clients = self.clients
        else:
            for c_id in client_ids:
                selected_clients.append(self.clients[c_id])
        # simulate cache bytes_model
        bytes_param = None
        for client in selected_clients:
            if bytes_param is None:
                bytes_param = client.flesh_global_model()
            else:
                client.flesh_global_model_lazy(bytes_param)
        # for client in selected_clients:
        #     client.flesh_global_model()

    def advance_train(self, client_ids=None):
        """
        advance one epoch train , let the clients local train and upload train infos
        """
        l.info('[advance one epoch]')
        selected_clients = []
        if client_ids is None:
            selected_clients = self.clients
        else:
            for c_id in client_ids:
                selected_clients.append(self.clients[c_id])
        for client in selected_clients:
            client.local_training()
            client.upload_train_info()

    def evaluate_local_model(self, client_ids=None):
        """
        evaluate the client's local trained model
        """
        l.info('[evaluate local model]')
        selected_clients = []
        if client_ids is None:
            selected_clients = self.clients
        else:
            for c_id in client_ids:
                selected_clients.append(self.clients[c_id])
        local_acc = []
        local_loss = []
        for client in selected_clients:
            acc, loss = client.evaluate(self.test_dl)
            local_acc.append(acc)
            local_loss.append(loss)
            l.info("round %d,client %d local model's accuracy is %.4f,loss is %.4f",
                   self.round,
                   client.id,
                   acc,
                   loss)
            # l.debug(f"model view {client.model_view()}")
        return local_acc, local_loss

    def evaluate_global_model(self):
        """
        evaluate the global aggregated model
        """
        l.info('[evaluate global model]')
        self.clients[0].flesh_global_model()
        acc, loss = self.clients[0].evaluate(self.test_dl)
        l.info("round %d,global accuracy %.4f,global loss %.4f", self.round, acc, loss)
        l.debug(f"model view {self.clients[0].model_view()}")
        return acc, loss

    def show_contributions(self):
        l.info("[show contribution]")
        cs = [self.round]
        for client in self.clients:
            c = client.invoker.get_contribution()
            cs.append(c)
        l.info(f"round {self.round},contributions:{cs}")
        self.contributions_result.append(cs)

    def advance_vote(self, client_ids=None):
        """
        advance one epoch vote process
        """
        l.info('[client vote]')
        selected_clients = []
        if client_ids is None:
            selected_clients = self.clients
        else:
            for c_id in client_ids:
                selected_clients.append(self.clients[c_id])
        for client in selected_clients:
            client.vote()

    def run(self, n=1, fixed_n_attacker=-1):
        # loop n times
        # the first round should upload init model,specific handle

        if self.round == 0:
            l.info(f'#######round {self.round}######')
            for i in range(self.n_client):
                self.clients[i].enroll()

            self.clients[0].init_model()
            # evaluate init model loss
            global_acc, global_loss = self.evaluate_global_model()
            round_result = [self.round, global_acc, global_loss] + [0 for _ in range(self.n_trainer)]
            self.train_result.append(round_result)
            self.round += 1
        np.random.seed((24 * self.round) % 125)
        for r in range(n):
            client_ids = None
            # number of attacker in participators is fixed
            if not fixed_n_attacker == -1:
                # number of normal node
                n_normal = self.n_client - self.n_attacker
                # number of normal node chosen
                fixed_n_normal = self.n_trainer - fixed_n_attacker
                order_attacker = np.random.permutation(self.n_attacker)[:fixed_n_attacker]
                order_normal = (np.random.permutation(n_normal) + self.n_attacker)[:fixed_n_normal]
                client_ids = np.hstack([order_attacker, order_normal])
                # random choice
            else:
                order = np.random.permutation(self.n_client)
                client_ids = order[0:self.n_trainer]

            l.info(f'#######round {self.round}#######')
            l.info(f'selected client id":{client_ids}')

            self.flesh(client_ids)
            # self.evaluate_local_model(client_ids)
            self.advance_train(client_ids)
            local_acc, _ = self.evaluate_local_model(client_ids)
            self.advance_vote(client_ids)
            self.show_contributions()
            global_acc, global_loss = self.evaluate_global_model()
            round_result = [self.round, global_acc, global_loss] + local_acc
            self.train_result.append(round_result)
            self.round += 1

    def save_model(self):
        self.clients[0].flesh_global_model()
        param_dict = self.clients[0].trainer.get_param_dict()

        pth_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time())) + \
                   '@nc{nc}_nt{nt}_na{na}@lr{lr}_bs{bs}_ep{ep}_{mn}_{ag}@r{r}.pth'.format(
                       nc=self.setting['train']['n_client'],
                       nt=self.setting['train']['n_trainer'],
                       na=self.setting['simulate']['n_attacker'],
                       lr=self.setting['train']['learning_rate'],
                       bs=self.setting['train']['batch_size'],
                       ep=self.setting['train']['epochs'],
                       mn=self.setting['task']['model_desc'],
                       ag=self.setting['train']['aggregate_method'],
                       r=self.round - 1)
        pth_path = os.path.join(self.setting['node']['model_dir'], pth_name)
        torch.save(param_dict, pth_path)
        l.info("save model param dict in pth path:{}".format(pth_path))

    def load(self, round, pth_name):
        self.round = round
        pth_path = os.path.join(self.setting['node']['model_dir'], pth_name)
        param_dict = torch.load(pth_path)
        self.clients[0].trainer.load_param_dict(param_dict)
        for i in range(self.n_client):
            self.clients[i].enroll()
        self.clients[0].init_model()
        global_acc, global_loss = self.evaluate_global_model()
        round_result = [self.round, global_acc, global_loss] + [0 for _ in range(self.n_trainer)]
        self.train_result.append(round_result)
        self.round += 1

    def save_result(self):
        df = pd.DataFrame(self.train_result,
                          columns=['round', 'global_acc', 'global_loss'] + [f'local_acc{i}' for i in
                                                                            range(self.n_trainer)],
                          dtype=float
                          )
        csv_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time())) + \
                   '@nc{nc}_nt{nt}_na{na}@lr{lr}_bs{bs}_ep{ep}_{mn}_{ag}.csv'.format(
                       nc=self.setting['train']['n_client'],
                       nt=self.setting['train']['n_trainer'],
                       na=self.setting['simulate']['n_attacker'],
                       lr=self.setting['train']['learning_rate'],
                       bs=self.setting['train']['batch_size'],
                       ep=self.setting['train']['epochs'],
                       mn=self.setting['task']['model_desc'],
                       ag=self.setting['train']['aggregate_method'])
        csv_path = os.path.join(self.setting['node']['results_dir'],
                                csv_name)
        df.to_csv(csv_path, index=False)
