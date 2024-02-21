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
   :widths: 5 5 7
   :header-rows: 0

   * - RL
     - RLTypes
     - Type
   * - Action
     - Discrete
     - int
   * - Action
     - Continuous
     - list[float]
   * - Action
     - Image
     - NDArray[np.uint8]
   * - Observation
     - Discrete
     - list[int]
   * - Observation
     - Continuous
     - NDArray[np.float32]
   * - Observation
     - Image
     - NDArray[np.float32]
  

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
