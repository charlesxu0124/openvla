import time
import cv2
import franka_env
import gym
import numpy as np
from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch
import absl.app
import absl.flags
from pynput import keyboard

FLAGS = absl.flags.FLAGS
absl.flags.DEFINE_string("primitive", "grasp", 'primitive to execute')

should_reset = False
def on_press(key):
    global should_reset
    try:
        if str(key) == 'Key.esc':
            should_reset = True
    except AttributeError:
        pass

def rollout(
        env,
        agent,
        max_path_length=np.inf,
        o=None,
        primitive='grasp',
        reset=True,
):
    global should_reset

    observations = []
    actions = []
    path_length = 0
    if reset:
        o = env.reset(gripper= 1 if primitive=='insert' else 0)
    else:
        o = o
    if primitive=='place_on_fixture' or primitive=='rotate':
        o, _, _, _ = env.step(np.array([0,0,0,0,0,0,1]))
    global should_reset
    should_reset = False
    if primitive=='insert':
        env.insertion_mode()

    while path_length < max_path_length:
        if should_reset:
            break
        #Get action from policy
        t = time.time()
        a = agent(o)
        print(f"inference time {time.time()-t}")
        print(a[-1])
        if a[6] > 0.5:
            a[6] = 1
        else:
            a[6] = 0
        # a[6] = 1
        a[:6] = a[:6] * env.action_space.high[:6]
        next_o, _, _, _= env.step(a)
        observations.append(o)
        actions.append(a)
        path_length += 1
        o = next_o
    env.freespace_mode()
    return dict(
        observations=observations,
        actions=actions,
    )


def main(_):
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()
    # Load Processor & VLA
    processor = AutoProcessor.from_pretrained("/home/panda/code/openvla/runs/openvla-7b+fmb_grasp_dataset+b16+lr-2e-05+lora-r32+dropout-0.0", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "/home/panda/code/openvla/runs/openvla-7b+fmb_grasp_dataset+b16+lr-2e-05+lora-r32+dropout-0.0", 
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to("cuda:0")

    def policy(obs):

        # Grab image input & format prompt
        image: Image.Image = Image.fromarray(obs['side_2']).resize((224, 224))
        cv2.imshow('image', np.array(image))
        cv2.waitKey(1)
        if FLAGS.primitive == 'grasp':
            prompt = "In: What action should the robot take to grasp the red object?\nOut:"
        elif FLAGS.primitive == 'insert':
            prompt = "In: What action should the robot take to insert the rectangle?\nOut:"
        elif FLAGS.primitive == 'place_on_fixture':
            prompt = "In: What action should the robot take to place the hexagon on the fixture?\nOut:"

        # Predict Action (7-DoF; un-normalize for FMB)
        inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
        action = vla.predict_action(**inputs, unnorm_key="fmb_dataset", do_sample=False)

        return action



    primitive = FLAGS.primitive
    if primitive == 'grasp' or primitive=='long_horizon':
        env = gym.make("Franka-FMB-v0", hz=10, start_gripper=0) 
        env.resetpos[:3] = np.array([0.45, 0.1, 0.22])
        env.reset_yaw=np.pi/2
        env.resetpos[3:] = env.euler_2_quat(np.pi, 0, env.reset_yaw)
    elif primitive == 'place_on_fixture':
        env = gym.make("Franka-FMB-v0", hz=10, start_gripper=1) 
        env.resetpos[:3] = np.array([0.45, 0.1, 0.03])
        env.reset_yaw=np.pi/2
        env.resetpos[3:] = env.euler_2_quat(np.pi, 0, env.reset_yaw)
    elif primitive=='rotate':
        env = gym.make("Franka-FMB-v0", hz=10, start_gripper=1) 
        env.resetpos[:3] = np.array([0.45, 0.1, 0.15])
        env.reset_yaw=np.pi/2
        env.resetpos[3:] = env.euler_2_quat(np.pi, 0, env.reset_yaw)
    elif primitive == 'regrasp':
        env = gym.make("Franka-FMB-v0", hz=10, start_gripper=0) 
        env.resetpos[:3] = np.array([0.6054150172032604,0.06086102586621338,0.1046158263847778])
        env.resetpos[3:] = np.array([0.6299743988034661,0.6971103396175138,-0.23715609242845734,0.24683277552754387])
    elif primitive == 'insert':
        env = gym.make("Franka-FMB-v0", hz=10, start_gripper=1) 
        # env.resetpos[:3] = np.array([0.5, -0.18, 0.22])
        # env.reset_yaw=np.pi/2
        # env.resetpos[3:] = env.euler_2_quat(np.pi, 0, env.reset_yaw)

        env.resetpos[:3] = np.array([0.5,-0.18,0.33])
        env.resetpos[3:] = np.array([0.7232287667289446,0.6900075685348428,-0.010295688497306624,-0.026901768319765842])
    else:
        raise ValueError("Unrecongized primitive name")

    for _ in range(10):
        rollout(env, policy, max_path_length=50, primitive=primitive, reset=True)

if __name__ == "__main__":
    absl.app.run(main)