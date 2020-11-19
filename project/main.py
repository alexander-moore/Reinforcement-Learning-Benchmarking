import argparse
from test import test
from environment import Environment
import agent_runner
import sys
import importlib

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
    env = Environment(env_name, args, atari_wrapper=True)

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
    args = parse()
    run(args)
