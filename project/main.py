import argparse
from test import test
from environment import Environment
import glob
import agent_runner
import sys
import importlib
import os
import re

def parse():
    parser = argparse.ArgumentParser(description="DS595/CS525 RL Project 3")
    parser.add_argument('--env_name', default=None, help='environment name')
    parser.add_argument('--train_dqn', action='store_true', help='whether train DQN')
    parser.add_argument('--test_dqn', action='store_true', help='whether test DQN')
    try:
        from argument import add_arguments
        parser = add_arguments(parser)
    except:
        pass
    args = parser.parse_args()
    return args

def list_agents_and_models():
    # Get all files in agents directory
    agent_files = glob.glob('./agents/*.py')

    print('')
    print('    Agents Available:')
    
    # print games we have implemented support for
    print('Breakout-v0')
    print('car_game, find env argument to make')

    # Get all classes in those files
    for af in agent_files:
        if '__init__' in af or 'replay_buffer' in af:
            continue
        
        # Get module name from file name
        _, file_name = os.path.split(af)
        file_name = file_name.replace('.py', '')

        # Get agent classes from file
        with open(af, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if re.match(r'^class', line):
                agent_class_name = (re.findall(r'^class\s+(\w+)\(.*', line))[0]
                print(f'        - {file_name}.{agent_class_name}')


    # Get all files in models directory
    model_files = glob.glob('./models/*.py')

    print('')
    print('    Models Available:')

    # Get all classes in those files
    for mf in model_files:
        if '__init__' in mf:
            continue
        
        # Get module name from file name
        _, file_name = os.path.split(mf)
        file_name = file_name.replace('.py', '')

        # Get agent classes from file
        with open(mf, 'r') as f:
            lines = f.readlines()
        for line in lines:
            if re.match(r'^class', line):
                model_class_name = (re.findall(r'^class\s+(\w+)\(.*', line))[0]
                print(f'        - {file_name}.{model_class_name}')
    print('')

def run(args):
    # Parse the model and environment args
    if not args.model:
        print('ERROR:  No model provided')
        sys.exit()

    if not args.agent:
        print('ERROR:  No agent provided.')
        sys.exit()

    agent_module, agent_class = args.agent.split('.')
    model_module, model_class = args.model.split('.')

    # Load the environment
    env_name = args.env_name or 'BreakoutNoFrameskip-v4'
    env = Environment(env_name, args, atari_wrapper=(not args.no_atari_wrapper))

    # Dynamically load an agent
    agents_module = importlib.import_module(f'agents.{agent_module}')
    agent_class = getattr(agents_module, agent_class)

    # Dynamically load a model
    models_module = importlib.import_module(f'models.{model_module}')
    model_class = getattr(models_module, model_class)

    # Instantiate a runner
    runner = agent_runner.AgentRunner(env, args, agent_class, model_class)

    if args.train_dqn:
        # Perform training
        runner.train()

    if args.test_dqn:
        # Perform testing
        runner.test()

if __name__ == '__main__':
    # Hackity hack hack
    if len(sys.argv) == 2:
        if sys.argv[1] == '--list':
            list_agents_and_models()
            sys.exit()

    args = parse()
    run(args)
