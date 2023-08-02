import srl
from srl.algorithms import ql

runner = srl.Runner("Grid", ql.Config())
runner.train(timeout=5)

runner.render_terminal()

"""
### 0, action 0(←), rewards[0.000]
env   {}
work0 {}
......
.   G.
. . X.
.P   .
......

 ←  : 0.26995
 ↓  : 0.27021
 →  : 0.22059
*↑  : 0.35530
### 1, action 3(↑), rewards[-0.040]
env   {}
work0 {'epsilon': 0}
......
.   G.
.P. X.
.    .
......

 ←  : 0.37256
 ↓  : 0.30609
 →  : 0.36743
*↑  : 0.46822
### 2, action 3(↑), rewards[-0.040]
env   {}
work0 {'epsilon': 0}
......
.P  G.
. . X.
.    .
......

 ←  : 0.47177
 ↓  : 0.42313
*→  : 0.59930
 ↑  : 0.51255
### 3, action 2(→), rewards[-0.040]
env   {}
work0 {'epsilon': 0}
......
. P G.
. . X.
.    .
......

 ←  : 0.51241
 ↓  : 0.64351
*→  : 0.76673
 ↑  : 0.59747
### 4, action 2(→), rewards[-0.040]
env   {}
work0 {'epsilon': 0}
......
.  PG.
. . X.
.    .
......

 ←  : 0.60441
 ↓  : 0.30192
*→  : 0.93905
 ↑  : 0.71430
### 5, action 2(→), rewards[1.000], done(env)
env   {}
work0 {}
......
.   P.
. . X.
.    .
......
"""
