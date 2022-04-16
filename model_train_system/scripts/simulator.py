import os
import sys
import copy
import logging
import pandas as pd
import numpy as np
import time

# sys.path.append('.')
# sys.path.append('..')
from brownie import accounts
from client_module.log import logger as l, init_logging
from scripts.deploy import deploy_contract
from scripts.setting import setting
import client_module.utils as u
from client_module.client import MockClient
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset


# aggregate_method = 'fed_vote_avg'

class ClusterSimulator(object):

    def __init__(self):
        init_logging(setting['log_dir'], log_level=logging.DEBUG)
        l.info(
            'aggregate_method = {am}, n_attacker = {na}, dataset = {ds}, model_name = {mn}, epochs= {ep}, n_vote= {nv}'.format(
                am=setting['aggregate_method'],
                na=setting['n_attacker'],
                ds=setting['dataset_description'],
                mn=setting['model_description'],
                ep=setting['epochs'],
                nv=setting['n_vote'],
            ))
        self.accounts = accounts
        self.n_client = len(self.accounts)
        self.n_attacker = setting['n_attacker']
        self.n_trainer = setting['n_trainer']
        self.n_vote = setting['n_vote']
        self.model_name = setting['model_description']

        # build and split dataset
        client_train_dataset, test_x_tensor, test_y_tensor = u.build_dataset(
            dataset_name=setting['dataset_description'],
            n_client=self.n_client,
            n_attacker=self.n_attacker,
            data_dir=setting['dataset_dir']
        )
        self.test_dl = DataLoader(
            TensorDataset(test_x_tensor, test_y_tensor),
            batch_size=5,
            shuffle=False
        )
        l.info(f"build test dataloader finish...")

        self.contract = deploy_contract()
        l.info(f"deploy contract,contract's info:{self.contract}")

        self.round = 0
        self.train_result = []

        self.clients = []
        for id in range(len(self.accounts)):
            client_setting = setting.copy()
            client_setting["model_name"] = self.model_name
            client_setting['aggregate_method'] = setting['aggregate_method']
            client_setting["ipfs_api"] = "/ip4/127.0.0.1/tcp/5001"
            train_x_tensor, train_y_tensor = client_train_dataset[id]
            client_setting["dataset"] = TensorDataset(
                train_x_tensor,
                train_y_tensor
            )
            client_setting['learning_rate'] = float(setting['learning_rate'])
            client_setting["id"] = id
            client_setting["account"] = self.accounts[id]
            client_setting["contract"] = self.contract

            new_client = MockClient(client_setting)
            self.clients.append(new_client)

        l.info(f"build {len(self.accounts)} clients success")

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
        for client in selected_clients:
            client.flesh_global_model()

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
            client.local_train()
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
            self.clients[0].init_model()
            # evaluate init model loss
            global_acc, global_loss = self.evaluate_global_model()
            round_result = [self.round, global_acc, global_loss] + [0 for _ in range(self.n_trainer)]
            self.train_result.append(round_result)
            self.round += 1
            n -= 1
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

            l.info(f'#######round {self.round}######')
            l.info(f'selected client id":{client_ids}')

            self.flesh(client_ids)
            self.evaluate_local_model(client_ids)
            self.advance_train(client_ids)
            local_acc, _ = self.evaluate_local_model(client_ids)
            self.advance_vote(client_ids)
            global_acc, global_loss = self.evaluate_global_model()
            round_result = [self.round, global_acc, global_loss] + local_acc
            self.train_result.append(round_result)
            self.round += 1

    def save_result(self):
        df = pd.DataFrame(self.train_result,
                          columns=['round', 'global_acc', 'global_loss'] + [f'local_acc{i}' for i in
                                                                            range(self.n_trainer)],
                          dtype=float
                          )
        csv_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime(time.time())) + \
                   '@npart{}_lr{}_bs{}_ep{}_{}_{}.csv'.format(setting['n_trainer'],
                                                              setting['learning_rate'],
                                                              setting['batch_size'],
                                                              setting['epochs'],
                                                              self.model_name,
                                                              setting['aggregate_method'])
        csv_path = os.path.join(setting['results_dir'],
                                csv_name)
        df.to_csv(csv_path, index=False)
