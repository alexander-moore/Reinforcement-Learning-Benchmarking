# CS525
Quincy and Brian and Alex and Scott and Jannik

Select environments from here:
https://gym.openai.com/envs/Breakout-v0/
https://gym.openai.com/envs/CarRacing-v0/

https://gym.openai.com/envs/#atari

Select some set of models from here:
https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html

- Develop framework such that environment and model can be selected at run-time with command line arugments
- Code should be correspondingly flexible to continuous and discrete action spaces, as well as different game types
- dqn_model.py can hold various model types appropriate for different games (likely not different agents)
- train_dqn.py likely should be agent-specific and environment-irreverent
