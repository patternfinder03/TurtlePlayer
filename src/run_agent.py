import argparse
import gymnasium as gym
from config import override_params, train_episodes
from agents.base_agent.baseagent import BaseAgent
from agents.dqn_agent.dqnagent_interval import DQNAgentInveral
from config import discount_factor, train_episodes, exploration_decay, learning_rate, memory_size, batch_size


def get_agent_class(agent_name, env):
    agent_params = {
        'env': env,
        'learning_rate': learning_rate,
        'discount_factor': discount_factor,
        'exploration_decay': exploration_decay,
        'memory_size': memory_size,
        'batch_size': batch_size
    }
    
    if agent_name == 'BaseAgent':
        return BaseAgent(env)
    elif agent_name == 'DQNInterval':
        return DQNAgentInveral(**agent_params)
    else:
        raise ValueError(f"Agent '{agent_name}' not recognized")

def parse_arguments():
    parser = argparse.ArgumentParser(description='Run Turtle Trading Strategy with specified agent and parameters.')
    parser.add_argument('--agent', type=str, default='BaseAgent', help='Specify the agent to use.')
    parser.add_argument('--all_stocks', action='store_true', default=False,help='Process all stocks if this flag is set.')
    parser.add_argument('--render', action='store_true', default=False,help='Render the environment if this flag is set.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_arguments()
    render_mode = 'human' if args.render else None
    
    if args.all_stocks:
        raise RuntimeError("This is outdated and needs to be updated")
        # from agents.run_multiple_stocks import run_all_stocks
        # agent = AgentClass(None)
        # run_all_stocks(agent, override_params)
    else:
        env = gym.make('TurtleTradingEnv', render_mode=render_mode, override_params=override_params)
        agent = get_agent_class(args.agent, env)
        from agents.run_single_stock import train_and_evaluate_agent_single_stock
        train_and_evaluate_agent_single_stock(agent, env, train_episodes)
