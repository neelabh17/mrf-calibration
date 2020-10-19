#!/usr/bin/env python
# coding: utf-8
#
# Author: Kazuto Nakashima
# URL:    https://kazuto1011.github.io
# Date:   07 January 2019

from __future__ import absolute_import, division, print_function

import click
import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from libs.datasets import get_dataset
from tqdm import tqdm
from libs.models import *
from libs.utils import DenseCRF
import os


def makedirs(dirs):
    if not os.path.exists(dirs):
        os.makedirs(dirs)


# matplotlib.use('TKAgg', force=True)


def get_device(cuda):
    cuda = cuda and torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    if cuda:
        current_device = torch.cuda.current_device()
        print("Device:", torch.cuda.get_device_name(current_device))
    else:
        print("Device: CPU")
    return device


def get_classtable(CONFIG):
    with open(CONFIG.DATASET.LABELS) as f:
        classes = {}
        for label in f:
            label = label.rstrip().split("\t")
            classes[int(label[0])] = label[1].split(",")[0]
    return classes


def setup_postprocessor(CONFIG):
    # CRF post-processor
    postprocessor = DenseCRF(
        iter_max=CONFIG.CRF.ITER_MAX,
        pos_xy_std=CONFIG.CRF.POS_XY_STD,
        pos_w=CONFIG.CRF.POS_W,
        bi_xy_std=CONFIG.CRF.BI_XY_STD,
        bi_rgb_std=CONFIG.CRF.BI_RGB_STD,
        bi_w=CONFIG.CRF.BI_W,
    )
    return postprocessor


def preprocessing(image, device, CONFIG):
    # Resize
    # scale = CONFIG.IMAGE.SIZE.TEST / max(image.shape[:2])
    # image = cv2.resize(image, dsize=None, fx=scale, fy=scale)
    raw_image = image.astype(np.uint8)

    # Subtract mean values
    image = image.astype(np.float32)
    image -= np.array(
        [
            float(CONFIG.IMAGE.MEAN.B),
            float(CONFIG.IMAGE.MEAN.G),
            float(CONFIG.IMAGE.MEAN.R),
        ]
    )

    # Convert to torch.Tensor and add "batch" axis
    image = torch.from_numpy(image.transpose(2, 0, 1)).float().unsqueeze(0)
    image = image.to(device)

    return image, raw_image


def inference(model, image, raw_image=None, postprocessor=None, temp=1):
    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    logits = logits / temp
    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()

    # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)

    return labelmap, probs


def inferenceTemp(model, image, raw_image=None, postprocessor=None, temp=1):
    _, _, H, W = image.shape

    # Image -> Probability map
    logits = model(image)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

    #  Performing temperature Scaling
    # print(logits[0][0][0][0])
    logits = logits / temp
    # print(logits[0][0][0][0])

    # print("--------------------------------")

    probs = F.softmax(logits, dim=1)[0]
    probs = probs.cpu().numpy()

    # # Refine the prob map with CRF
    if postprocessor and raw_image is not None:
        probs = postprocessor(raw_image, probs)

    labelmap = np.argmax(probs, axis=0)
    probmap = np.max(probs, axis=0)

    return labelmap, probmap, probs


@click.group()
@click.pass_context
def main(ctx):
    """
    Demo with a trained model
    """

    print("Mode:", ctx.invoked_subcommand)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="PyTorch model to be loaded",
)
@click.option(
    "-i",
    "--image-path",
    type=click.Path(exists=True),
    required=True,
    help="Image to be processed",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
@click.option("--crf", is_flag=True, show_default=True, help="CRF post-processing")
def single(config_path, model_path, image_path, cuda, crf):
    """
    Inference from a single image
    """

    # Setup
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    classes = get_classtable(CONFIG)
    postprocessor = setup_postprocessor(CONFIG) if crf else None

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    # Inference
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image, raw_image = preprocessing(image, device, CONFIG)
    # labelmap,probmap = inference(model, image, raw_image, postprocessor)
    labelmap, probmap = inference(model, image, raw_image, postprocessor=None)
    labelmapt, probmapt = inference(
        model, image, raw_image, postprocessor=None, temp=1.2367
    )
    labels = np.unique(labelmap)
    probmap = np.max(probmap, axis=0)
    probmapt = np.max(probmapt, axis=0)
    print(probmap.shape)
    # import pdb; pdb.set_trace()

    # Show result for each class
    rows = np.floor(np.sqrt(len(labels) + 1))
    cols = np.ceil((len(labels) + 1) / rows)

    plt.figure(figsize=(10, 10))
    ax = plt.subplot(rows, cols, 1)
    ax.set_title("Input image")
    ax.imshow(raw_image[:, :, ::-1])
    ax.axis("off")

    for i, label in enumerate(labels):
        mask = labelmap == label
        maskt = labelmapt == label
        # import pdb; pdb.set_trace()
        ax = plt.subplot(rows, cols, i + 2)
        ax.set_title(classes[label])
        ax.imshow(raw_image[..., ::-1])
        # probapbilisticaly determines the fading
        # ax.imshow(mask.astype(np.float32), alpha=mask.astype(np.float32)*(probmap-probmapt)/0.08)
        ax.imshow(
            mask.astype(np.float32),
            alpha=(probmap - probmapt) / np.max(probmap - probmapt),
        )
        # ax.imshow(maskt.astype(np.float32), alpha=(maskt.astype(np.float32)*probmapt)/2)
        # ax.imshow(mask.astype(np.float32), alpha=0.5)
        # ax.imshow((mask==False).astype(np.float32), alpha=0.5)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("output.png")


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="PyTorch model to be loaded",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
@click.option("--crf", is_flag=True, show_default=True, help="CRF post-processing")
@click.option("--camera-id", type=int, default=0, show_default=True, help="Device ID")
def live(config_path, model_path, cuda, crf, camera_id):
    """
    Inference from camera stream
    """

    # Setup
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.set_grad_enabled(False)
    torch.backends.cudnn.benchmark = True

    classes = get_classtable(CONFIG)
    postprocessor = setup_postprocessor(CONFIG) if crf else None

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    # UVC camera stream
    cap = cv2.VideoCapture(camera_id)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"YUYV"))

    def colorize(labelmap):
        # Assign a unique color to each label
        labelmap = labelmap.astype(np.float32) / CONFIG.DATASET.N_CLASSES
        colormap = cm.jet_r(labelmap)[..., :-1] * 255.0
        return np.uint8(colormap)

    def mouse_event(event, x, y, flags, labelmap):
        # Show a class name of a mouse-overed pixel
        label = labelmap[y, x]
        name = classes[label]
        print(name)

    window_name = "{} + {}".format(CONFIG.MODEL.NAME, CONFIG.DATASET.NAME)
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    while True:
        _, frame = cap.read()
        image, raw_image = preprocessing(frame, device, CONFIG)
        labelmap = inference(model, image, raw_image, postprocessor)
        colormap = colorize(labelmap)

        # Register mouse callback function
        cv2.setMouseCallback(window_name, mouse_event, labelmap)

        # Overlay prediction
        cv2.addWeighted(colormap, 0.5, raw_image, 0.5, 0.0, raw_image)

        # Quit by pressing "q" key
        cv2.imshow(window_name, raw_image)
        if cv2.waitKey(10) == ord("q"):
            break


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="PyTorch model to be loaded",
)
@click.option(
    "-d",
    "--image-dir",
    type=click.Path(exists=True),
    required=True,
    help="Image Folder to be processed",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
@click.option("--crf", is_flag=True, show_default=True, help="CRF post-processing")
def evalcrf(config_path, model_path, image_dir, cuda, crf):
    """
    Inference from a single image
    """

    # Setup
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    classes = get_classtable(CONFIG)
    print(crf)
    # postprocessor = setup_postprocessor(CONFIG) if crf else None
    postprocessor = setup_postprocessor(CONFIG)

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    # Inference
    images = os.listdir(image_dir)
    saveDir = "eval_crf_worst"
    makedirs(saveDir)

    import pickle

    f1 = open("cal_acc.pickle", "rb")
    caldata = pickle.load(f1)
    f1.close()

    f1 = open("uncal_acc.pickle", "rb")
    uncaldata = pickle.load(f1)
    f1.close()

    caldata = dict(caldata)
    uncaldata = dict(uncaldata)

    common = []
    for item in caldata:
        common.append([item, caldata[item] - uncaldata[item]])

    common.sort(key=lambda x: x[1])
    # import pdb;pdb.set_trace()

    # Dataset
    dataset = get_dataset(CONFIG.DATASET.NAME)(
        root=CONFIG.DATASET.ROOT,
        split=CONFIG.DATASET.SPLIT.VAL,
        ignore_label=CONFIG.DATASET.IGNORE_LABEL,
        mean_bgr=(CONFIG.IMAGE.MEAN.B, CONFIG.IMAGE.MEAN.G, CONFIG.IMAGE.MEAN.R),
        augment=False,
    )

    myDataset={}
    for i in tqdm(range(len(dataset))):
        image_id, image, gt_label = dataset.__getitem__(i)
        myDataset[image_id]=i
    def get_gt_with_id(image_id):
        image_id2, img_raw, _gt_label = dataset.__getitem__(myDataset[image_id])
        return _gt_label
    def get_img_with_id(image_id):
        image_id2, img_raw, _gt_label = dataset.__getitem__(myDataset[image_id])
        return img_raw
    
    # Path to logit files
    logit_dir = os.path.join(
        CONFIG.EXP.OUTPUT_DIR,
        "features",
        CONFIG.EXP.ID,
        CONFIG.MODEL.NAME.lower(),
        CONFIG.DATASET.SPLIT.VAL,
        "logit",
    )

    for img_no, (imageName, loss) in enumerate(common[:30]):
        
        print(f"At Image : {img_no}", end="\r")

        image_id, image, gt_label = dataset.__getitem__(myDataset[imageName])
        print(image.shape)

        filename = os.path.join(logit_dir, image_id + ".npy")
        logit = np.load(filename)
        # import pdb; pdb.set_trace()

        _, H, W = image.shape
        logit = torch.FloatTensor(logit)[None, ...]
        logit = F.interpolate(logit, size=(H, W), mode="bilinear", align_corners=False)
        
        # Temperature scaling the factor of 1.2367 was precalculated 
        # T=1.2367
        T=1
        uncal_logit=logit/T

        uncal_prob = F.softmax(uncal_logit, dim=1)[0].numpy()
        image = image.astype(np.uint8).transpose(1, 2, 0)
        print(image.shape)
        uncal_before_postprocess=uncal_prob.copy()
        uncal_prob = postprocessor(image, uncal_prob)


        uncal_label = np.argmax(uncal_prob, axis=0)
        uncal_mask = (uncal_label == gt_label)
        uncal_correct = uncal_mask.sum()
        uncal_total = uncal_mask.shape[0] * uncal_mask.shape[1]
        uncal_acc = uncal_correct / uncal_total
        
        T=1.2367
        cal_logit=logit/T

        cal_prob = F.softmax(cal_logit, dim=1)[0].numpy()
        # image = image.astype(np.uint8).transpose(1, 2, 0)
        cal_prob = postprocessor(image, cal_prob)


        cal_label = np.argmax(cal_prob, axis=0)
        cal_mask = (cal_label == gt_label)
        cal_correct = cal_mask.sum()
        cal_total = cal_mask.shape[0] * cal_mask.shape[1]
        cal_acc = cal_correct / cal_total
        
        print(cal_acc-uncal_acc)
    
        image_path = os.path.join(image_dir, imageName + ".jpg")
        _image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        _image, raw_image = preprocessing(_image, device, CONFIG)
        
        # postprocessor_image=get_img_with_id(imageName)
        # postprocessor_image = postprocessor_image.astype(np.uint8).transpose(1, 2, 0)


        # uncal_labelmap, uncal_probmap, uncal_all_class_prob = inferenceTemp(
        #     model, image, postprocessor_image, postprocessor, temp=1
        # )
        # cal_labelmap, cal_probmap, cal_all_class_prob = inferenceTemp(
        #     model, image, postprocessor_image, postprocessor, temp=1.2367
        # )
        # uncal_labels = np.unique(uncal_labelmap)
        # cal_labels = np.unique(cal_labelmap)

        uncal_labels = np.unique(uncal_label)
        cal_labels = np.unique(cal_label)
        
        
        # gt_label=get_gt_with_id(imageName)
        # if(np.sum((cal_labelmap!=uncal_labelmap).astype(np.float32))==0):
        if False:
            pass
        else:
            print("yay")
            # Show result for each class
            cols = int(np.ceil((max(len(uncal_labels), len(cal_labels)) + 1)))+1
            rows = 4
            
            

            plt.figure(figsize=(20, 20))
            ax = plt.subplot(rows, cols, 1)
            ax.set_title("Input image")
            ax.imshow(raw_image[:, :, ::-1])
            ax.axis("off")
            ax = plt.subplot(rows, cols, cols + 1)
            ax.set_title("Accuracy dif = {:0.3f}".format(loss))
            ax.imshow(raw_image[:, :, ::-1])
            ax.axis("off")
            # ax = plt.subplot(rows, cols, 2 * cols + 1)
            # gradient = np.linspace(0, 1, 256)
            # gradient = np.vstack((gradient, gradient))
            # ax.imshow(gradient, cmap="nipy_spectral")
            # ax.set_title("Acc")
            # ax.imshow(raw_image[:, :, ::-1])
            # ax.axis("off")

            for i, label in enumerate(uncal_labels):
                ax = plt.subplot(rows, cols, i + 3)
                ax.set_title("Uncalibrated-" + classes[label])
                ax.imshow(uncal_prob[label], cmap="nipy_spectral")

                ax.axis("off")

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, cols + i + 3)
                ax.set_title("Calibrated-" + classes[label])
                ax.imshow(cal_prob[label], cmap="nipy_spectral")
                ax.axis("off")
            # for i, label in enumerate(uncal_labels):
            #     ax = plt.subplot(rows, cols, 4*cols + i + 3)
            #     ax.set_title("Calibrated-" + classes[label])
            #     ax.imshow(uncal_before_postprocess[label], cmap="nipy_spectral")
            #     ax.axis("off")

            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, 2 * cols + i + 3)
                
                min_dif=np.min(uncal_prob[label] - cal_prob[label])
                max_dif=np.max(uncal_prob[label] - cal_prob[label])

                dif_map=np.where((uncal_prob[label] - cal_prob[label])>0,(uncal_prob[label] - cal_prob[label]),0)
                
                ax.set_title(
                    "decrease: "
                    + classes[label]
                    + " max={:0.3f}".format(
                        max_dif
                    )
                )
                ax.imshow(
                    dif_map
                    / max_dif,
                    cmap="nipy_spectral",
                )
                ax.axis("off")
            
            for i, label in enumerate(cal_labels):
                ax = plt.subplot(rows, cols, 3 * cols + i + 3)
                
                min_dif=np.min(uncal_prob[label] - cal_prob[label])
                max_dif=np.max(uncal_prob[label] - cal_prob[label])

                dif_map=np.where((uncal_prob[label] - cal_prob[label])<0,(uncal_prob[label] - cal_prob[label]),0)
                
                ax.set_title(
                    "increase: "
                    + classes[label]
                    + " max={:0.3f}".format(
                        -min_dif
                    )
                )
                ax.imshow(
                    dif_map
                    / min_dif,
                    cmap="nipy_spectral",
                )
                ax.axis("off")
            
            acc_cal_mask=(gt_label==cal_label).astype(np.float32)
            acc_uncal_mask=(gt_label==uncal_label).astype(np.float32)
            # for i, label in enumerate(cal_labels):
            ax = plt.subplot(rows, cols, 2 * cols + 1)

            maxi=np.max(acc_uncal_mask - acc_cal_mask)
            un=np.unique(acc_uncal_mask - acc_cal_mask)
            print(un)
            dif_map=np.where((acc_uncal_mask - acc_cal_mask)<0,(acc_uncal_mask - acc_cal_mask),0)
            
            ax.set_title(
                "increase ac: "
                + classes[label]
                + " max={:0.3f}".format(
                    1
                )
            )
            ax.imshow(
                dif_map/(-1),
                cmap="nipy_spectral",
            )

            # for i, label in enumerate(cal_labels):
            ax = plt.subplot(rows, cols, 3 * cols + 1)
            dif_map=np.where((acc_uncal_mask - acc_cal_mask)>0,(acc_uncal_mask - acc_cal_mask),0)
            
            ax.set_title(
                "decrease ac: "
                + classes[label]
                + " max={:0.3f}".format(
                    1
                )
            )
            ax.imshow(
                dif_map/1,
                cmap="nipy_spectral",
            )
                
            
            

            plt.tight_layout()
            save_file = os.path.join(saveDir, imageName+f"_{30-img_no}-30")
            plt.savefig(save_file)


@main.command()
@click.option(
    "-c",
    "--config-path",
    type=click.File(),
    required=True,
    help="Dataset configuration file in YAML",
)
@click.option(
    "-m",
    "--model-path",
    type=click.Path(exists=True),
    required=True,
    help="PyTorch model to be loaded",
)
@click.option(
    "-d",
    "--image-dir",
    type=click.Path(exists=True),
    required=True,
    help="Image Folder to be processed",
)
@click.option(
    "--cuda/--cpu", default=True, help="Enable CUDA if available [default: --cuda]"
)
@click.option("--crf", is_flag=True, show_default=True, help="CRF post-processing")
def evalcalibration(config_path, model_path, image_dir, cuda, crf):
    """
    Inference from a single image
    """

    # Setup
    CONFIG = OmegaConf.load(config_path)
    device = get_device(cuda)
    torch.set_grad_enabled(False)

    classes = get_classtable(CONFIG)
    postprocessor = setup_postprocessor(CONFIG) if crf else None

    model = eval(CONFIG.MODEL.NAME)(n_classes=CONFIG.DATASET.N_CLASSES)
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    print("Model:", CONFIG.MODEL.NAME)

    # Inference
    images = os.listdir(image_dir)
    saveDir = "eval_calib_images"
    makedirs(saveDir)

    for img_no, imageName in enumerate(images):
        print(f"At Image : {img_no}", end="\r")
        image_path = os.path.join(image_dir, imageName)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image, raw_image = preprocessing(image, device, CONFIG)
        uncal_labelmap, uncal_probmap = inferenceTemp(
            model, image, raw_image, postprocessor=False, temp=1
        )
        cal_labelmap, cal_probmap = inferenceTemp(
            model, image, raw_image, postprocessor=False, temp=1.2367
        )
        uncal_labels = np.unique(uncal_labelmap)
        cal_labels = np.unique(cal_labelmap)

        # Show result for each class
        rows = np.floor(np.sqrt(len(uncal_labels) + 1))
        cols = np.ceil((len(uncal_labels) + 1) / rows)

        plt.figure(figsize=(10, 10))
        ax = plt.subplot(rows, cols, 1)
        ax.set_title(
            "Input image(max dif={:0.4}".format(np.max(uncal_probmap - cal_probmap))
        )
        ax.imshow(raw_image[:, :, ::-1])
        ax.axis("off")

        for i, label in enumerate(uncal_labels):
            mask = uncal_labelmap == label
            ax = plt.subplot(rows, cols, i + 2)
            ax.set_title("Uncalibrated-" + classes[label])
            ax.imshow(raw_image[..., ::-1])
            ax.imshow(
                mask.astype(np.float32),
                alpha=(uncal_probmap - cal_probmap)
                / np.max(uncal_probmap - cal_probmap),
            )
            ax.axis("off")

        plt.tight_layout()
        save_file = os.path.join(saveDir, imageName)
        plt.savefig(save_file)


if __name__ == "__main__":
    main()
