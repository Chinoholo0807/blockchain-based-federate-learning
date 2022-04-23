scripts to start cluster simulator

## activate env
conda activate deeplearning_common
## use console
brownie console

## import ClusterSimulator
from scripts.simulator import ClusterSimulator
cs = ClusterSimulator('conf/simulate.yaml')

# start simulate
cs.run(10)

# save the result
cs.save_result()
cs.save_model()
```