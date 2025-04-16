import srl
from srl.algorithms import ql

runner = srl.Runner("Grid", ql.Config())
runner.train(timeout=5)

runner.render_terminal()

"""
### 0
state  : 1,3
action : None
rewards:[0.000]
total rewards:[0.000]
env   {}
work0 {'epsilon': 0.1}
......
.   G.
. . X.
.P   .
......

 0(←)  : 0.31790
 1(↓)  : 0.30021
 2(→)  : 0.22420
*3(↑)  : 0.40947

### 1
state  : 1,2
action : 3(↑)
rewards:[-0.040]
total rewards:[-0.040]
env   {}
work0 {'epsilon': 0}
......
.   G.
.P. X.
.S   .
......

 0(←)  : 0.41061
 1(↓)  : 0.30757
 2(→)  : 0.37957
*3(↑)  : 0.51605

### 2
state  : 1,2
action : 3(↑)
rewards:[-0.040]
total rewards:[-0.080]
env   {}
work0 {'epsilon': 0}
......
.   G.
.P. X.
.S   .
......

 0(←)  : 0.41061
 1(↓)  : 0.30757
 2(→)  : 0.37957
*3(↑)  : 0.51605

### 3
state  : 1,1
action : 3(↑)
rewards:[-0.040]
total rewards:[-0.120]
env   {}
work0 {'epsilon': 0}
......
.P  G.
. . X.
.S   .
......

 0(←)  : 0.50963
 1(↓)  : 0.47168
*2(→)  : 0.63577
 3(↑)  : 0.52524

### 4
state  : 2,1
action : 2(→)
rewards:[-0.040]
total rewards:[-0.160]
env   {}
work0 {'epsilon': 0}
......
. P G.
. . X.
.S   .
......

 0(←)  : 0.55444
 1(↓)  : 0.65966
*2(→)  : 0.76021
 3(↑)  : 0.64972

### 5
state  : 3,1
action : 2(→)
rewards:[-0.040]
total rewards:[-0.200]
env   {}
work0 {'epsilon': 0}
......
.  PG.
. . X.
.S   .
......

 0(←)  : 0.65326
 1(↓)  : 0.51852
*2(→)  : 0.95872
 3(↑)  : 0.78929
......
.   P.
. . X.
.S   .
......


### 6, done()
state  : 4,1
action : 2(→)
rewards:[1.000]
total rewards:[0.800]
env   {}
work0 {'epsilon': 0}
"""
