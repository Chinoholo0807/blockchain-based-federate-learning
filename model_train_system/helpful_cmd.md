scripts to start cluster simulator

## activate env
```py
conda activate deeplearning_common
```
## use console
```py
brownie console
```
then input cmd in the brownie console
## ClusterMode
### init ClusterSimulator
```py
from scripts.simulator import ClusterSimulator
cs = ClusterSimulator('conf/cluster.yaml')
```
###  start simulate
```py
# run 10 global iter
cs.run(10)
```
### save the result
```
# save the acc\loss result
cs.save_result()
# save the model param
cs.save_model()
```
## DecentralMode
### init contract
```py
from scripts.deploy import deploy_contract_yaml
m = deploy_contract_yaml('conf/decentral.yaml')
```
### init node
```py
from scripts.node import Node
n = Node('conf/node1.yaml')
n.run()
```
