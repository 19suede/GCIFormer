import torch
from utils import select_model
import argparse
import os
from config import select_config
import numpy as np
import h5py
import scipy.io as sio
import time

map_dict = {
    "QuickBird": "qb",
    "WorldView3": "wv3",
    "GaoFen2": "gf2",
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="WorldView3", required=False, help="QuickBird, WorldView3, GaoFen2")
    parser.add_argument("--model", default="Base1", required=False, help="Base1, ...")
    parser.add_argument("--test_mode", default="f", required=False, help="r for reduced resolution or f for full resolution")
    parser.add_argument("--cuda_id", default=1, type=int, required=False)
    parser.add_argument("--prenorm", default=False, type=bool, required=False)
    parser.add_argument("--epoch", default=100, required=False)

    parser.add_argument("--save_checkpoint_path", default="saved_models", type=str, required=False)
    parser.add_argument("--save_fused_img_path", default="fused_imgs", type=str, required=False)
    parser.add_argument("--seed", default=1234, type=int, required=False)

    args = parser.parse_args()
    return args


def load_test_data(args, config):
    if args.test_mode == "r":
        with h5py.File(config.testset_path_r, 'r') as file:
            pan_data = np.array(file['pan'])
            lms_data = np.array(file['lms'])
            ms_data = np.array(file['ms'])
            gt_data = np.array(file['gt'])
            if args.prenorm:
                pan_data = pan_data / config.dr
                lms_data = lms_data / config.dr
                ms_data = ms_data / config.dr

        return pan_data, lms_data, ms_data, gt_data

    else:
        with h5py.File(config.testset_path_f, 'r') as file:
            pan_data = np.array(file['pan'])
            lms_data = np.array(file['lms'])
            ms_data = np.array(file['ms'])
            if args.prenorm:
                pan_data = pan_data / config.dr
                lms_data = lms_data / config.dr
                ms_data = ms_data / config.dr

        return pan_data, lms_data, ms_data


def save_to_mat(i, fused, args):
    if args.test_mode == "r":
        data = {"sr": fused.transpose(1, 2, 0)}
        save_fused_img_path = args.save_fused_img_path + f"/{args.dataset}/{args.model}/RR"
        if not os.path.exists(save_fused_img_path):
            os.makedirs(save_fused_img_path)
        sio.savemat(os.path.join(save_fused_img_path, f"output_Test(HxWxC)_{map_dict[args.dataset]}_data{i + 1}.mat"),
                    data)
    else:
        data = {"sr": fused.transpose(1, 2, 0)}
        save_fused_img_path = args.save_fused_img_path + f"/{args.dataset}/{args.model}/FR"
        if not os.path.exists(save_fused_img_path):
            os.makedirs(save_fused_img_path)
        sio.savemat(
            os.path.join(save_fused_img_path, f"output_Test(HxWxC)_{map_dict[args.dataset]}_data_fr{i + 1}.mat"), data)


def main():
    args = get_args()
    config = select_config(args.dataset)
    args.seed = args.seed
    device = f"cuda:{args.cuda_id}" if torch.cuda.is_available() else "cpu"

    """
    load test data
    """
    if args.test_mode == "r":
        pan_data, lms_data, ms_data, gt_data = load_test_data(args, config)
    else:
        pan_data, lms_data, ms_data = load_test_data(args, config)

    bs = pan_data.shape[0]

    """
    load model
    """
    model = select_model(args.model, config)
    model = model.to(device)
    model_info = f"{args.model}" + \
                 f"_{config.num_blocks}_{config.patch_size}_{config.window_size}_{config.model_dim}_{config.hidden_ch}" + \
                 f"_cudaid{args.cuda_id}"

    model_path = os.path.join(args.save_checkpoint_path, args.dataset, model_info)
    checkpoint = torch.load(os.path.join(model_path, f"epoch{args.epoch}.pth"), map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        for i in range(bs):
            pan = torch.tensor(pan_data[i]).to(torch.float32).to(device).unsqueeze(0)
            lms = torch.tensor(lms_data[i]).to(torch.float32).to(device).unsqueeze(0)
            ms = torch.tensor(ms_data[i]).to(torch.float32).to(device).unsqueeze(0)

            fused, _ = model(pan, lms, ms)

            if args.prenorm:
                fused = torch.clamp(fused, 0, 1)
                fused = fused * config.dr

            fused = fused[0]
            fused = fused.detach().cpu()
            fused = np.array(fused)

            save_to_mat(i, fused, args)


if __name__ == "__main__":
    main()
