import utils
import pytest
from brownie import ComputingMarket,accounts

setting = {
        "batchSize":20,
        "learningRate":"0.01",
        "epochs":7,
        "nParticipators":10,
    }
@pytest.fixture
def market():
    setting_list = [v for v in setting.values()]
    contract =  ComputingMarket.deploy("Test Model Name",setting_list,{"from": accounts[0]})
    return contract

def test_market_init(market):
    market_setting = market.setting()
    require_setting_list = [v for v in setting.values()]
    for i,s in enumerate(market_setting):
        assert s == require_setting_list[i]
    
def test_upload_single_localupdate(market):
    version = 0
    training_size = 10
    update_gradient = b'laksdhfjagnfbvmnsddlkhweoihtasfkjasfsdfmcxvb'
    uploader = accounts[0]
    market.uploadLocalUpdate(version,training_size,update_gradient,{"from":uploader})
    updates = market.getLocalUpdates(version)
    assert len(updates) ==1
    _uploader, _training_size,_version,_gradient_hex= updates[0]
    
    assert uploader == _uploader
    assert training_size == _training_size
    assert version == _version
    assert update_gradient == utils.hex2bytes(_gradient_hex)

def test_upload_2account_localupdate(market):
    version = 0
    training_size = 10
    update_gradient = b'laksdhfjagnfbvmnsddlkhweoihtasfkjasfsdfmcxvb'
    uploader = accounts[0]

    updates = market.getLocalUpdates(version)
    assert len(updates) ==1
    market.uploadLocalUpdate(version,training_size,update_gradient,{"from":uploader})
    updates = market.getLocalUpdates(version)
    assert len(updates) ==1
    _uploader, _training_size,_version,_gradient_hex= updates[0]
    
    assert uploader == _uploader
    assert training_size == _training_size
    assert version == _version
    assert update_gradient == utils.hex2bytes(_gradient_hex)

    uploader = accounts[1]
    market.uploadLocalUpdate(version,training_size,update_gradient,{"from":uploader})
    updates = market.getLocalUpdates(version)
    assert len(updates) ==2