from gym.envs.registration import register

register(
    id='vision-v0',
    entry_point='gym_vision.envs:VisionEnv',
)

