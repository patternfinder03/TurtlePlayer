from gymnasium.envs.registration import register

register(
    id='TurtleTradingEnv',
    entry_point='gym_turtle_env.turtle_env:TurtleTradingEnv',
)