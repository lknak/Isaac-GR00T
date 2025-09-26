from datasets import load_dataset

import json, copy
import pandas as pd, numpy as np
import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

import subprocess

schema = pa.schema([
    ("observation.state", pa.list_(pa.float32())),  # list of floats
    ("action", pa.list_(pa.float32())),             # list of floats
    ("timestamp", pa.float64()),                    # numeric timestamp
    ("annotation.human.action.task_description", pa.int64()),
    ("task_index", pa.int64()),
    ("annotation.human.validity", pa.int64()),
    ("episode_index", pa.int64()),
    ("index", pa.int64()),
    ("next.reward", pa.float32()),
    ("next.done", pa.bool_())
])

BASE_PATH = "/home/lux/.cache/huggingface/hub/datasets--unitreerobotics--G1_Dex3_BlockStacking_Dataset/snapshots/57faa2cf516e008f96d91fe3b67ad53a74f012e6"

# Load the dataset directly
dataset = load_dataset("unitreerobotics/G1_Dex3_BlockStacking_Dataset")

# Convert each split to JSONL
for split in dataset.keys():
    # dataset[split].to_json(f"{split}.jsonl")

    with open(f'{split}.jsonl', 'r') as json_file:
        json_list = list(json_file)

    t = {0: "stack three block", 1: "valid"}
    d = {"episode_index": 0, "tasks": [], "length": 0}
    out_l = {}

    out_full = {}

    for json_str in tqdm.tqdm(json_list):
        result = json.loads(json_str)

        ep = result["episode_index"]
        ti = result['task_index']
        if ep in out_l.keys():
            out_l[ep]["length"] += 1
            if ti not in out_l[ep]["tasks"]:
                out_l[ep]["tasks"].insert(-1, ti)
            out_full[ep].append(result)
        else:
            out_l[ep] = {
                "length": 0,
                "episode_index": ep,
                "tasks": [ti, 1]
            }
            out_full[ep] = [result]
    
    with open(f'{split}_episodes.jsonl', 'w') as json_file:
        json.dump(list(out_l.values()), json_file)
    

    processes = []
    # split episodes to parquet
    for ep, d in tqdm.tqdm(out_full.items()):
        records = [{
                        "observation.state": d_["observation.state"],
                        "action": d_["action"],
                        "timestamp": d_["timestamp"],
                        "annotation.human.action.task_description": d_["task_index"],  # mirror task_index
                        "task_index": d_["task_index"],
                        "annotation.human.validity": 1,  # hardcoded
                        "episode_index": d_["episode_index"],
                        "index": d_["index"],
                        "next.reward": int(i >= len(d) - 1),
                        "next.done": i >= len(d) - 1,
                    }
                    for i, d_ in enumerate(d)]
        table = pa.Table.from_pydict({
                k: [r[k] for r in records] for k in records[0]
            }, schema=schema)
        pq.write_table(table, f"{BASE_PATH}/data/chunk-000/file-{str(d[0]['episode_index']).zfill(3)}.parquet")

        cmd = [

            "ffmpeg",
            "-ss", str(records[0]['index'] * 1 / 30),
            "-i", f"{BASE_PATH}/videos/observation.images.cam_right_high/chunk-000/all_videos.mp4",
            "-t", str((records[-1]['index'] - records[0]['index']) * 1 / 30),
            "-c:v", "libx264",
            "-preset", "ultrafast",
            f"{BASE_PATH}/videos/observation.images.cam_right_high/chunk-000/episode_{str(d[0]['episode_index']).zfill(3)}.mp4"
        ]

        # cmd = [
        #     "ffmpeg", 
        #     "-i", f"{BASE_PATH}/videos/observation.images.cam_right_high/chunk-000/all_videos.mp4",
        #     "-vf", f"select='between(n\,{records[0]['index']}\,{records[-1]['index']})',setpts=N/30/TB",
        #     "-vsync", "vfr",
        #     f"{BASE_PATH}/videos/observation.images.cam_right_high/chunk-000/episode_{str(d[0]['episode_index']).zfill(3)}.mp4"
        # ]
    
        processes.append(subprocess.Popen(cmd))

        if len(processes) > 16:
            processes[0].wait()
            del processes[0]
        
        # frames = get_frames_by_indices(f"",
        #                          [r["index"] for r in records],
        #                          "decord")
        # # h264 codec encoding
        # torchvision.io.write_video(f"", 
        #                            frames, 30, video_codec="av1")