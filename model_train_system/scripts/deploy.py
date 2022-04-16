#!/usr/bin/python3


from brownie import ModelTrain, accounts
from scripts.setting import train_setting,task_setting


def deploy_contract():
    train = [v for v in train_setting.values()]
    task = [v for v in task_setting.values()]
    contract = ModelTrain.deploy(
        train,
        task,
        {"from": accounts[0]}
    )
    return contract
