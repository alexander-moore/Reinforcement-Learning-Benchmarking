def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--model', type=str, required=True, help='The model to exercise:\ndqn_model.DQN')
    parser.add_argument('--agent', type=str, required=True, help='The agent to exercise:\nagent_dqn.Agent_DQN')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', type=float, default=1.5e-4, help='Learning rate for training')
    parser.add_argument('--replay_buffer_size', type=int, default=10000, help='Size of the replay buffer')
    parser.add_argument('--replay_buffer_batch_size', type=int, default=32, help='Size of the mini-batch samples pulled from replay buffer')
    parser.add_argument('--min_buffer_size', type=int, default=5000, help='Minimum size of buffer that must be populated before training begins.')
    parser.add_argument('--epsilon', type=float, default=1.0, help='Starting epsilon value')
    parser.add_argument('--epsilon_min', type=float, default=0.025, help='Final value for epsilon after decaying is complete.')
    parser.add_argument('--epsilon_steps', type=int, default=1000000, help='Number of steps until epsilon decaying completes.')
    parser.add_argument('--gamma', type=float, default=0.99, help='Gamma for how much we weight our target Q values.')
    parser.add_argument('--num_episodes', type=int, default=10000, help='Number of episodes to run for training.')
    parser.add_argument('--start_episode', type=int, default=1, help='Used for restarting training from a checkpoint to indicate episode to start from.  Mostly for accounting purposes.')
    parser.add_argument('--render_train', action='store_true', help='Indicates to show GUI rendering of training episodes being played.')
    parser.add_argument('--render_test', action='store_true', help='Indicates to show GUI rendering of test episodes being played.')
    parser.add_argument('--target_network_update_interval', type=int, default=5000, help='Number of steps that occur until the target network is updated.')
    parser.add_argument('--model_prefix', type=str, default='', help='If present, then this prefix is added to the default model name.')
    parser.add_argument('--archive_dir', type=str, default='archive', help='Indicates where to store data collections.')
    parser.add_argument('--run_name', type=str, required=True, help='Provide a name for this execution for archive collection name.')
    parser.add_argument('--test_cycle_interval', type=int, default=1000, help='Perform a test cycle in intervals of this value.')
    parser.add_argument('--model_path', type=str, default=None, help='Path to model weights to load.')
    parser.add_argument('--optimizer_path', type=str, default=None, help='Path to optimizer weights to load.')

    return parser
