# Applied Inverse Reinforcement Learning from Observation on the Unitree A1 Robot
These are the launcher files for my bachelor thesis. They run the IRL algorithms GAIL and VAIL on a MuJoCo model of the Unitree A1. The whole setting is implemented in the mushroomrl library. Replay agents replays the learned agent to evaluate the performance. In the last commit are functions to observe the feet height and to plot the reward of the replayed agents. That will only work if you also use the adapted mushroomrl library. For further details have a look at my thesis.

The required dataset are at https://drive.google.com/drive/folders/1w5SeejITkFCH0KgEUUuaGRv6MRdIiPFo?usp=sharing.

2023_02_23_19_48_33_straight is a dataset with 50.000 datapoints per trajectory. There are 3 trajectories of the robot walking forward. Each with a different random seed and noise

2023_02_23_19_48_33 is a dataset with 50.000 datapoints per trajectory and 24 trajectories in 8 direction with 3 different seeds of noise per direction.

2023_02_23_19_22_49 is the same but with less datapoints (5.000) per trajectory for quicker testing.
