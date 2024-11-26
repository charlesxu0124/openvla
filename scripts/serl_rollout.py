import json
import os
from pathlib import Path
import time
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
# from serl_experiments.mappings import CONFIG_MAPPING
from serl_experiments.connector_insert.config import TrainConfig as ConnectorInsertTrainConfig
from serl_experiments.vla_place.config import TrainConfig as PickPlaceConfig
from serl_experiments.fmb_insert.config import TrainConfig as InsertConfig
from serl_experiments.fmb_grasp_move.config import TrainConfig as FMBConfig 

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
    o, _ = env.reset()
    for _ in range(max_path_length):
        # Get action from policy
        a = agent(o)
        next_o, rew, done, truncated, info = env.step(a)
        rew = int(info['original_state_obs']['tcp_pose'][2] < 0.11)
        done = done or rew
        observations.append(o)
        actions.append(a)
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
    # checkpoint_path = "/media/nvmep3p/openvla_checkpoints/openvla-7b+connector_insert_human_dataset+b6+lr-2e-05+lora-r32+dropout-0.0+wrist_1_45_traj/step-25000"
    # checkpoint_path = "/media/nvmep3p/openvla_checkpoints/openvla-7b+cucumber_pick_place_rl_dataset+b6+lr-2e-05+lora-r32+dropout-0.0+wrist_50_traj/step-200000"
    # checkpoint_path = "/media/nvmep3p/openvla_checkpoints/openvla-7b+pepper_pick_place_human_dataset+b6+lr-2e-05+lora-r32+dropout-0.0+wrist_50_traj/step-200000"
    # checkpoint_path = "/media/nvmep3p/openvla_checkpoints/openvla-7b+fmb_human_composition_dataset+b6+lr-2e-05+lora-r32+dropout-0.0+wrist_75_relabeled_human_traj/step-200000"
    checkpoint_path = "/media/nvmep3p/openvla_checkpoints/openvla-7b+fmb25_human_insert_dataset+b2+lr-2e-05+lora-r32+dropout-0.0+wrist/step-50000"
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
        with open(Path(checkpoint_path).parent / "dataset_statistics.json", "r") as f:
            vla.norm_stats = json.load(f)

    def policy(obs):
        # Grab image input & format prompt
        image = obs["wrist_1"][0]
        prompt = "In: What action should the robot take to insert the double square object? "
        # Predict Action (7-DoF; un-normalize for FMB)``
        image: Image.Image = Image.fromarray(image)
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

        action = vla.predict_action(**inputs, unnorm_key="fmb25_human_insert_dataset", do_sample=True)
        # action = vla.predict_action(**inputs, unnorm_key="cucumber_pick_place_rl_dataset", do_sample=True)
        action = action[:6]
        return action

    config = InsertConfig()
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
    for n in range(30):
        start_time = time.time()
        _, rew = rollout(env, policy, max_path_length=200)
        finish_time = time.time()
        if rew:
            cycle_times.append(finish_time - start_time)
            success_count += 1
        print(f"Success Rate: {success_count} / {n+1}")
        print(f"Average Cycle Time: {np.mean(cycle_times)}")


if __name__ == "__main__":
    absl.app.run(main)
