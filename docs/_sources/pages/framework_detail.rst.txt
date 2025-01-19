.. _framework_detail:

=====================
Detailed Framework
=====================


Play flow
----------------

.. image:: ../../diagrams/playflow1.png

・1 Episode loop

.. image:: ../../diagrams/playflow2.png


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

+ SpaceBase(srl.base.spaces)

.. list-table::
   :header-rows: 0

   * - Class
     - Type
     - SpaceType
   * - DiscreteSpace
     - int
     - DISCRETE
   * - ArrayDiscreteSpace
     - list[int]
     - DISCRETE
   * - ContinuousSpace
     - float
     - CONTINUOUS
   * - ArrayContinuousSpace
     - list[float]
     - CONTINUOUS
   * - BoxSpace
     - NDArray[AnyType]
     - srl.base.define.SpaceTypes
   * - MultiSpace
     - list[SpaceBase]
     - MULTI


+ RL type

.. list-table::
   :header-rows: 0
  
   * - 
     - Action
     - Observation
     - Observation window
   * - Discrete
     - | int  
       | DiscreteSpace
     - | list[int]
       | ArrayDiscreteSpace
     - | list[int]
       | ArrayDiscreteSpace
   * - Continuous
     - | list[float]
       | ArrayContinuousSpace
     - | NDArray[np.float32]
       | BoxSpace
     - | NDArray[np.float32]
       | BoxSpace
   * - Image
     - | NDArray[np.uint8]
       | BoxSpace
     - | NDArray[np.float32]
       | BoxSpace
     - | NDArray[np.float32]
       | BoxSpace
