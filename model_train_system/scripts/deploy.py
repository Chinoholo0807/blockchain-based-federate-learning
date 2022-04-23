#!/usr/bin/python3
from brownie import ModelTrain, accounts
from scripts.helpful_scripts import read_yaml

def deploy_contract_yaml(yaml_path):
    setting = read_yaml(yaml_path)
    contract = deploy_contract_setting(setting)
    print('deploy contract ',contract)
    return contract


def deploy_contract_setting(setting):
    train = (
        setting['train']['batch_size'],
        setting['train']['learning_rate'],
        setting['train']['epochs'],
        setting['train']['n_trainer'],
        setting['train']['n_poll'],
        setting['train']['n_client'],
        setting['train']['max_version'],
    )
    task = (
        setting['task']['task_desc'],
        setting['task']['model_desc'],
        setting['task']['dataset_desc'],
    )
    contract = ModelTrain.deploy(
        train,
        task,
        {"from": accounts[0]}
    )
    return contract
