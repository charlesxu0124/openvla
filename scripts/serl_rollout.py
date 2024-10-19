import json
import os
from pathlib import Path
import pickle
import time
import cv2
import gymnasium as gym
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
from franka_env.envs.franka_fmb_env import DefaultFMBEnvConfig
import franka_env
from serl_experiments import config
# from serl_experiments.mappings import CONFIG_MAPPING
from serl_experiments.connector_insert.config import TrainConfig as ConnectorInsertTrainConfig
from serl_experiments.vla_place.config import TrainConfig as VLAPlaceTrainConfig

import torch
import absl.app
import absl.flags

def rollout(
    env,
    agent,
    max_path_length=np.inf,
    o=None,
):
    observations = []
    actions = []
    path_length = 0
    o, _ = env.reset()
    while path_length < max_path_length:
        # Get action from policy
        a = agent(o)
        # print(a[-1])
        next_o, rew, done, truncated, info = env.step(a)
        observations.append(o)
        actions.append(a)
        path_length += 1
        o = next_o
        if done:
            print("Reward: ", rew)
            break
    return dict(
        observations=observations,
        actions=actions,
    ), rew


def main(_):
    # checkpoint_path = "/media/nvmep3p/openvla_checkpoints/openvla-7b+fmb75_dslsr_insert_dataset+b2+lr-2e-05+lora-r32+dropout-0.0+wrist_1/step-200000"
    # checkpoint_path = "/media/nvmep3p/openvla_checkpoints/openvla-7b+vga_insert_human_dataset+b4+lr-2e-05+lora-r32+dropout-0.0+wrist_1/step-125000"
    checkpoint_path = "/media/nvmep3p/openvla_checkpoints/openvla-7b+vga_insert_rl_dataset+b4+lr-2e-05+lora-r32+dropout-0.0+wrist_1/step-50000"
    # checkpoint_path = "openvla/openvla-7b"

    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained(checkpoint_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        checkpoint_path,
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to("cuda:0")

    if os.path.isdir(checkpoint_path):
        with open(Path(checkpoint_path) / "dataset_statistics.json", "r") as f:
            vla.norm_stats = json.load(f)

    def policy(obs):
        # Grab image input & format prompt
        image = obs["wrist_1"][0]
        prompt = "In: What action should the robot take to insert the VGA connector?"
        # Predict Action (7-DoF; un-normalize for FMB)``
        image: Image.Image = Image.fromarray(image)
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

        action = vla.predict_action(**inputs, unnorm_key="vga_insert_rl_dataset", do_sample=True)
        action = action[:6]
        return action

    config = ConnectorInsertTrainConfig()
    env = config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
    )

    # trigger jit
    policy(env.observation_space.sample())
    policy(env.observation_space.sample())

    success_count = 0
    cycle_times = []
    for n in range(20):
        start_time = time.time()
        _, rew = rollout(env, policy, max_path_length=50)
        finish_time = time.time()
        if rew:
            cycle_times.append(finish_time - start_time)
            success_count += 1
        print(f"Success Rate: {success_count} / {n+1}")
        print(f"Average Cycle Time: {np.mean(cycle_times)}")


if __name__ == "__main__":
    absl.app.run(main)
