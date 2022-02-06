# MADDPG

Simple pytorch implementation of [MADDPG](https://arxiv.org/pdf/1706.02275) for an cooperative task with a shared actor and critic network.

## Train

Run 
```
python train.py
```
while being in the src folder.
The parameters can be changed by either creating a new config file and then
```
python train.py --cfg <config-file path>
```
or directly i.e.
```
python train.py --lr_critic 0.01
```
which takes the parameters from the default file but ovverrides the given parameter/s.

## Test

To test the performance of the agent, run
```
python test.py --run_folder <path to the run-folder>
```
to test and render the performance of the newest saved agent weights.
