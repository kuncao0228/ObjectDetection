import argparse
import logging
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import pdb
from unet import UNet
from utils.data_vis import plot_img_and_mask
from utils.data_bayes import get_stats
from utils.dataset import BasicDataset
import cv2
import glob
from tqdm import tqdm


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)
    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        probs = probs.cpu().numpy()
        probs = np.transpose(probs,(1,2,0))
        probs = cv2.resize(probs,(img.shape[2],img.shape[3]))
        probs = np.transpose(probs,(2,0,1))

        # pdb.set_trace()
        full_mask = probs
    return full_mask > out_threshold


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--model', '-m', default='checkpoints/CP_epoch50.pth',
                        metavar='FILE',
                        help="Specify the file in which the model is stored")
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+',
                        help='filenames of input images', required=True)

    parser.add_argument('--output', '-o', metavar='INPUT', nargs='+',
                        help='Filenames of ouput images')
    parser.add_argument('--viz', '-v', action='store_true',
                        help="Visualize the images as they are processed",
                        default=False)
    parser.add_argument('--no-save', '-n', action='store_true',
                        help="Do not save the output masks",
                        default=True)
    parser.add_argument('--mask-threshold', '-t', type=float,
                        help="Minimum probability value to consider a mask pixel white",
                        default=0.5)
    parser.add_argument('--scale', '-s', type=float,
                        help="Scale factor for the input images",
                        default=1)
    parser.add_argument('--petri','-p',default=False)

    return parser.parse_args()


def get_output_filenames(args):
    in_files = args.input
    out_files = []

    if not args.output:
        for f in in_files:
            pathsplit = os.path.splitext(f)
            out_files.append("{}_OUT{}".format(pathsplit[0], pathsplit[1]))
    elif len(in_files) != len(args.output):
        logging.error("Input files and output files are not of the same length")
        raise SystemExit()
    else:
        out_files = args.output

    return out_files


def mask_to_image(mask):
    print(type(mask))
    # return Image.fromarray(mask)
    return Image.fromarray((mask[2,:,:] * 255).astype(np.uint8))

def get_pngs(input):
    return glob.glob(input+'/*.png')

if __name__ == "__main__":
    args = get_args()
    in_files = get_pngs(args.input[0])
    print(in_files)
    out_files = get_output_filenames(args)

    net = UNet(n_channels=3, n_classes=4)

    logging.info("Loading model {}".format(args.model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))

    logging.info("Model loaded !")

    correct = 0
    stats = []
    for i, fn in enumerate(tqdm(in_files)):
        print (fn)
        logging.info("\nPredicting image {} ...".format(fn))

        img = Image.open(fn)
        img1 = cv2.imread(fn,0)
        mask = predict_img(net=net,
                           full_img=img,
                           scale_factor=args.scale,
                           out_threshold=args.mask_threshold,
                           device=device)

        if not args.no_save:
            out_fn = out_files[i]
            result = mask_to_image(mask)
            result.save(out_files[i])

            logging.info("Mask saved to {}".format(out_files[i]))

        logging.info("Visualizing results for image {}, close to continue ...".format(fn))
        print (args.petri)
        try:
            if args.petri:
                correct += plot_img_and_mask(img1, mask, fn)
            else:
                print("Here")
                stats.append(get_stats(img1,mask,fn))
        except:
            continue
    true_positives = 0
    total_positives = 0
    predicted_positives = 0
    over_ocr_true = 0
    over_ocr_total = 0
    for stat in stats:
        (tp,tp_fn,tp_fp,ocr_true,ocr_total) = stat
        true_positives+=tp
        total_positives+=tp_fn
        predicted_positives+=tp_fp
        over_ocr_true += ocr_true
        over_ocr_total += ocr_total
        if tp==tp_fn and tp==tp_fp:
            correct+=1

    filename = (args.input[0]).split("/")[-1]+".txt"
    file = open("results/"+filename, 'w')


    file.write("Accuracy:"+str(100*correct/(i+1))+"%")
    file.write("Precision:"+str(100*true_positives/(predicted_positives))+"%")
    file.write("Recall:"+str(100*true_positives/(total_positives))+"%")
    file.write("Ocr_accuracy:"+str(100*over_ocr_true/(over_ocr_total))+"%")

    file.close()
