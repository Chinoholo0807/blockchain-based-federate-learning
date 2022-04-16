import json
from web3 import Web3

w3 = Web3(Web3.HTTPProvider("HTTP://127.0.0.1:7545"))
print(w3.isConnected())
print('------------------------')
print('accounts:',w3.eth.accounts)
print('------------------------')
address = '0xee963fd60b36Cd5F986A7e1FcBCe6BF2d4E07B37'
abi_filename = '/Users/bytedance/blockchian/federate-learning-computing-market/federate_learning_with_blockchain/on_chain/dapp-demo/build/contracts/TestContract.json'

with open(abi_filename) as f:
    data = json.load(f)
    global abi
    abi = data.get('abi')

contract = w3.eth.contract(address=address,abi = abi)

print(contract)
w3.eth.default_account = w3.eth.accounts[-1]
tx_hash = contract.functions.set('alice','bob').transact()
w3.eth.waitForTransactionReceipt(tx_hash)
print(contract.functions.get('alice').call())

# build a txn
