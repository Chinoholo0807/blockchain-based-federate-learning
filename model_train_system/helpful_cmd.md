scripts to start cluster simulator

## activate env
conda activate deeplearning_common
## use console
brownie console
## ClusterMode
### init ClusterSimulator
from scripts.simulator import ClusterSimulator

cs = ClusterSimulator('conf/cluster.yaml')

###  start simulate
cs.run(10)

### save the result
cs.save_result()
cs.save_model()

## DecentralMode
### init contract
from scripts.deploy import deploy_contract_yaml
m = deploy_contract_yaml('conf/decentral.yaml')
### init node
from scripts.node import Node
n = Node('conf/node1.yaml')
n.run()