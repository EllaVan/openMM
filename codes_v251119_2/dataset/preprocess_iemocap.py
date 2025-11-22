import argparse
import json
import glob
import logging
import os
import pickle
import random

import pandas as pd
import soundfile as sf
import tqdm
import numpy as np
import torch
# from moviepy.editor import VideoFileClip
from moviepy.video.io.VideoFileClip import VideoFileClip
from sklearn.model_selection import train_test_split


SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

LABEL_MAP = {
    "hap": 0,
    "sad": 1,
    "sur": 2,
    "dis": 3,
    'ang': 4,
    'fea': 5, 
    'fru': 6, 
    'exc': 7,
    'neu': 8,
}
# anger, happy, sad, neutral, frustrated, excited, fear, surprise, disgust

logging.getLogger().setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def export_mp4_to_audio(
    mp4_file: str,
    wav_file: str,
    verbose: bool = False,
):
    """Convert mp4 file to wav file

    Args:
        mp4_file (str): Path to mp4 input file
        wav_file (str): Path to wav output file
        verbose (bool, optional): Whether to print ffmpeg output. Defaults to False.
    """
    try:
        video = VideoFileClip(mp4_file)
    except:
        logging.warning(f"Failed to load {mp4_file}")
        return 0
    audio = video.audio
    audio.write_audiofile(wav_file, verbose=verbose)
    return 1


def preprocess_IEMOCAP(args):
    data_root = '/media/sda/wf/openMM/Datasets/IEMOCAP/IEMOCAP_withVideo/IEMOCAP_full_release'
    ignore_length = 0

    session_id = list(range(1, 6))

    samples = []
    labels = []
    iemocap2label = LABEL_MAP
    iemocap2label.update({"exc": 1})

    for sess_id in tqdm.tqdm(session_id):
        sess_path = os.path.join(data_root, "Session{}".format(sess_id))
        sess_autio_root = os.path.join(sess_path, "sentences/wav")
        sess_text_root = os.path.join(sess_path, "dialog/transcriptions")
        sess_label_root = os.path.join(sess_path, "dialog/EmoEvaluation")
        label_paths = glob.glob(os.path.join(sess_label_root, "*.txt"))
        for l_path in label_paths:
            l_name = os.path.basename(l_path)
            transcripts_path = os.path.join(sess_text_root, l_name)
            with open(transcripts_path, "r") as f:
                transcripts = f.readlines()
                transcripts = {
                    t.split(":")[0]: t.split(":")[1].strip() for t in transcripts
                }
            with open(l_path, "r") as f:
                label = f.read().split("\n")
                for l in label:
                    if str(l).startswith("["):
                        data = l[1:].split()
                        wav_folder = data[3][:-5]
                        wav_name = data[3] + ".wav"
                        emo = data[4]
                        wav_path = os.path.join(sess_autio_root, wav_folder, wav_name)
                        wav_data, _ = sf.read(wav_path, dtype="int16")
                        # Ignore samples with length < ignore_length
                        if len(wav_data) < ignore_length:
                            logging.warning(
                                f"Ignoring sample {wav_path} with length {len(wav_data)}"
                            )
                            continue
                        emo = iemocap2label.get(emo, None)
                        text_query = data[3] + " [{:08.4f}-{:08.4f}]".format(
                            float(data[0]), float(data[2][:-1])
                        )
                        if emo is not None:
                            text = transcripts.get(text_query, None)
                            if text is None:
                                text_query = data[3] + " [{:08.4f}-{:08.4f}]".format(
                                    float(data[0]), (float(data[2][:-1]) + 0.0001)
                                )
                                text = transcripts.get(text_query, None)
                                if text is None:
                                    text_query = data[
                                        3
                                    ] + " [{:08.4f}-{:08.4f}]".format(
                                        float(data[0]) + 0.0001, float(data[2][:-1])
                                    )
                                    text = transcripts.get(text_query, None)
                                    if text is None:
                                        print(transcripts.keys())
                                        print(text_query)
                                        raise Exception
                            samples.append((wav_path, text, emo))
                            labels.append(emo)

    # Shuffle and split
    temp = list(zip(samples, labels))
    random.Random(args.seed).shuffle(temp)
    samples, labels = zip(*temp)
    train, test_samples, train_labels, _ = train_test_split(
        samples, labels, test_size=0.1, random_state=args.seed
    )
    train_samples, val_samples, _, _ = train_test_split(
        train, train_labels, test_size=0.1, random_state=args.seed
    )
    # Save data
    os.makedirs(args.dataset + "_preprocessed", exist_ok=True)
    with open(os.path.join(args.dataset + "_preprocessed", "train.pkl"), "wb") as f:
        pickle.dump(train_samples, f)
    with open(os.path.join(args.dataset + "_preprocessed", "val.pkl"), "wb") as f:
        pickle.dump(val_samples, f)
    with open(os.path.join(args.dataset + "_preprocessed", "test.pkl"), "wb") as f:
        pickle.dump(test_samples, f)

    logging.info(f"Train samples: {len(train_samples)}")
    logging.info(f"Val samples: {len(val_samples)}")
    logging.info(f"Test samples: {len(test_samples)}")
    logging.info(f"Saved to {args.dataset + '_preprocessed'}")
    logging.info("Preprocessing finished successfully")

def main(args):
    preprocess_fn = {
        "IEMOCAP": preprocess_IEMOCAP,
    }

    preprocess_fn[args.dataset](args)


def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ds", "--dataset", type=str, default="IEMOCAP", choices=["IEMOCAP"]
    )
    # parser.add_argument(
    #     "-dr",
    #     "--data_root",
    #     default='/media/sda/wf/openMM/Datasets/IEMOCAP/IEMOCAP_withVideo/IEMOCAP_full_release',
    #     type=str,
    #     help="Path to folder containing IEMOCAP data",
    #     required=True,
    # )
    parser.add_argument("--all_classes", default=True, action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--ignore_length",
        type=int,
        default=0,
        help="Ignore samples with length < ignore_length",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    main(args)
