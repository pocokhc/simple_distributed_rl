.. _framework_detail:

=====================
Detailed Framework
=====================


Play flow
----------------

.. image:: ../../diagrams/playflow.png


Multiplay flow
----------------

.. image:: ../../diagrams/overview-multiplay.drawio.png


Distributed flow
----------------

.. image:: ../../diagrams/runner_distributed_flow.png


Class diagram
----------------

+ RL

.. image:: ../../diagrams/class_rl.png

+ Env

.. image:: ../../diagrams/class_env.png

+ Run

.. image:: ../../diagrams/class_runner.png


Interface Type
----------------

+ Env type

.. list-table::
   :widths: 10 10
   :header-rows: 0

   * - Env
     - Type
   * - Action
     - SpaceBase
   * - Observation
     - SpaceBase


+ RL type

.. list-table::
   :widths: 5 5 5 5 5
   :header-rows: 0

   * - 
     - Discrete
     - Continuous
     - Image
     - Multiple
   * - Action
     - int
     - list[float]
     - NDArray[np.uint8]
     - list[RL type]
   * - Observation
     - list[int]
     - NDArray[np.float32]
     - NDArray[np.float32]
     - list[NDArray[np.float32]]
  

+ SpaceBase(srl.base.env.spaces)

.. list-table::
   :widths: 20 10
   :header-rows: 0

   * - Class
     - Type
   * - DiscreteSpace
     - int
   * - ArrayDiscreteSpace
     - list[int]
   * - ContinuousSpace
     - float
   * - ArrayContinuousSpace
     - list[float]
   * - BoxSpace
     - NDArray[AnyType]
   * - ArraySpace
     - list[SpaceBase]
