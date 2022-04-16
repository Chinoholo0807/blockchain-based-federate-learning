scripts to start cluster simulator
```
# activate env
conda activate deeplearning_common
# use console
brownie console

# import ClusterSimualtor
from scripts.simulator import ClusterSimulator
cs = ClusterSimulator()

# access specific client in cluster

# start simulate
cs.run(10)

# save the result
cs.save_result()
```