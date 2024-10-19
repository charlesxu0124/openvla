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
from serl_experiments.vla_usb_insert.config import TrainConfig as VLAUSBInsertTrainConfigs
from serl_experiments.vla_place.config import TrainConfig as VLAPlaceTrainConfig

import torch
import absl.app
import absl.flags
from pynput import keyboard
from franka_env.envs.wrappers import (
    GripperCloseEnv,
    SpacemouseIntervention,
)

FLAGS = absl.flags.FLAGS
absl.flags.DEFINE_string("primitive", "insert", "primitive to execute")
absl.flags.DEFINE_string("exp_name", "vla_place", "experiment name")

should_reset = False


def on_press(key):
    global should_reset
    try:
        if str(key) == "Key.esc":
            should_reset = True
    except AttributeError:
        pass


def rollout(
    env,
    agent,
    max_path_length=np.inf,
    o=None,
    primitive="grasp",
    reset=True,
):
    global should_reset

    observations = []
    actions = []
    path_length = 0
    if reset:
        o, _ = env.reset(gripper=1 if primitive == "insert" else 0)
    else:
        o = o
    if primitive == "place_on_fixture" or primitive == "rotate":
        o, _, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, 1]))
    global should_reset
    should_reset = False

    while path_length < max_path_length:
        if should_reset:
            break
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
    checkpoint_path = "/media/nvmep3p/openvla_checkpoints/openvla-7b+rl_hexagon_place_dataset:1.0.0+b8+lr-2e-05+lora-r32+dropout-0.0+global/step-100000"
    # checkpoint_path = "openvla/openvla-7b"

    listener = keyboard.Listener(on_press=on_press)
    listener.start()
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
        image = obs["side"][0]
        # if FLAGS.primitive == "grasp":
        #     prompt = "In: What action should the robot take to pick up the large short red hexagon lying horizontally?\nOut:"
        # elif FLAGS.primitive == "insert":
        #     prompt = "In: What action should the robot take to insert the Display Port connector?\nOut:"
        # elif FLAGS.primitive == "place_on_fixture":
        #     prompt = "In: What action should the robot take to place the hexagon on the fixture?\nOut:"
        prompt = "In: What action should the robot take to place the hexagon on the plate"
        # prompt = "In: What action should the robot take to place the green pepper on the plate"
        # Predict Action (7-DoF; un-normalize for FMB)``
        image: Image.Image = Image.fromarray(image)
        # breakpoint()
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

        action = vla.predict_action(**inputs, unnorm_key="rl_hexagon_place_dataset:1.0.0", do_sample=True)
        # action = action[:6]
        return action

    primitive = FLAGS.primitive

    # if primitive == "insert":
    #     class EnvConfig(DefaultFMBEnvConfig):
    #         REALSENSE_CAMERAS = {
    #             "wrist_1": "127122270350",
    #             "side_1": "128422270679",
    #         }
    #         ABS_POSE_LIMIT_LOW = np.array([0.45, -0.28, 0.06, np.pi - 0.05, -0.05, np.pi/4])
    #         ABS_POSE_LIMIT_HIGH = np.array([0.7, -0.1, 0.25, np.pi + 0.05, 0.05, 3*np.pi/4])
    #         RANDOM_RESET = False
    #         RANDOM_XY_RANGE = 0.03
    #         RANDOM_RZ_RANGE = np.pi / 35
    #         RESET_POSE = np.array([0.5, -0.18, 0.25, np.pi, 0, np.pi/2])
    #         MAX_EPISODE_LENGTH = 100
    #         COMPLIANCE_PARAM = {
    #             "translational_stiffness": 2000,
    #             "translational_damping": 89,
    #             "rotational_stiffness": 150,
    #             "rotational_damping": 7,
    #             "translational_Ki": 10,
    #             "rotational_Ki": 0,
    #             "translational_clip_x": 0.007,
    #             "translational_clip_y": 0.007,
    #             "translational_clip_z": 0.005,
    #             "translational_clip_neg_x": 0.007,
    #             "translational_clip_neg_y": 0.007,
    #             "translational_clip_neg_z": 0.005,
    #             "rotational_clip_x": 0.04,
    #             "rotational_clip_y": 0.04,
    #             "rotational_clip_z": 0.04,
    #             "rotational_clip_neg_x": 0.04,
    #             "rotational_clip_neg_y": 0.04,
    #             "rotational_clip_neg_z": 0.04,
    #         }
    #         IMAGE_RESOLUTION: int = 224
    #         IMAGE_CROP = {
    #             "wrist_1": lambda image: image[150:450, 200:500],
    #             "wrist_2": lambda image: image,
    #             "side_1": lambda image: image[0:350, 150: 400],
    #             "side_2": lambda image: image,
    #         }
    # env = gym.make("Franka-FMB-v1",
    #                 fake_env=False,
    #                 save_video=False,
    #                 config=EnvConfig,
    #         )

    config = VLAPlaceTrainConfig()
    env = config.get_environment(
        fake_env=False,
        save_video=False,
        classifier=False,
    )

    success_count = 0
    for n in range(100):
        _, rew = rollout(env, policy, max_path_length=50, primitive=primitive, reset=True)
        success_count += rew
        print(f"Success Rate: {success_count} / {n+1}")

if __name__ == "__main__":
    absl.app.run(main)
