import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 3개와 4번 GPU 선택
#os.add_dll_directory("C:\\Users\\sosal\\.jupyter\\Harvard\\openslide-win64-20220811\\bin")
import openslide
import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle
import cv2
import gc
from itertools import product
import time
import random
import glob
import umap
import argparse

import matplotlib.pyplot as plt
from PIL import Image
from skimage import io, transform

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader


#from MIL.models.CLAM import CLAM_SB, CLAM_MB
#from MIL.utils import MIL_generator, Metrics, WSIutils
import large_image

from matplotlib.colors import ListedColormap
cmap = plt.cm.jet
cmap_jet = cmap(np.arange(cmap.N))
cmap_jet[:1, -1] = 0  # 첫번째 색상을 투명하게 설정
cmap_jet = ListedColormap(cmap_jet)

cmap_patch = np.array([[0, 0, 0, 1], [0, 0, 0, 0]])  # Black & Transparent
cmap_patch = ListedColormap(cmap_patch)

from timm.models.vision_transformer import VisionTransformer

import timm
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
from huggingface_hub import login, hf_hub_download
login('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')

def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",}
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url

def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", 16)
    print("patch_size:", patch_size)
    model = VisionTransformer(img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress))
        print(verbose)
    return model

class PatchDataset(Dataset):
    def __init__(self, SlideName, patch_size=448, stride=448, downsample=2, transform=None):
        self.slide = openslide.open_slide(SlideName)
        self.Xmax = self.slide.dimensions[0]
        self.Ymax = self.slide.dimensions[1]
        self.magnification = float(self.slide.properties['aperio.AppMag']) # should be 40
        print("Magnification:", self.magnification)
        self.stride = stride
        self.patch_size = patch_size
        self.downsample = downsample
        self.transform = transform
        self.Patches, self.Coords = self.Patchify()
        
    def Patchify(self):
        Patches, Coords = [], []
        for Ypos in tqdm(range(0, self.Ymax, self.stride)):
            image = self.slide.read_region(location=(0, Ypos), level=0, size=(self.Xmax, self.patch_size)) # Read All X in Ypos
            image = image.resize( (int(image.size[0]/self.downsample), int(self.patch_size/self.downsample)) )
            image = np.array(image)[:, :, :3]
            for Xpos in range(0, image.shape[1], int(self.stride/self.downsample)):
                Patch = image[:, Xpos:Xpos+int(self.patch_size/self.downsample), :]
                if self.isWhitePatch(Patch)+self.isBlackPatch(Patch)+self.isWrongSizePatch(Patch)==0:
                    Patches.append(Patch)
                    Coords.append([Xpos*2, Ypos])
        return Patches, Coords
    
    def isWhitePatch(self, patch, satThresh=7):
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        return True if np.mean(patch_hsv[:,:,1]) < satThresh else False

    def isBlackPatch(self, patch, rgbThresh=40):
        return True if np.all(np.mean(patch, axis = (0,1)) < rgbThresh) else False

    def isWrongSizePatch(self, patch): # Good: false, problem: True
        patch_size = int(self.patch_size / self.downsample)
        return True if patch.shape != (patch_size, patch_size, 3) else False
    
    def __len__(self):
        return len(self.Patches)

    def __getitem__(self, index):
        ret_patch = self.Patches[index]
        if self.transform:
            return self.transform(ret_patch), self.Coords[index]
        return ret_patch, self.Coords[index]
    # PatchGenerator.transform = 
    

def DrawUmapHeatmap(Barcode, Feature, Coord, patch_size, cmap_patch=cmap_patch, cmap_jet=cmap_jet,
                    HeatmapDownsample = 14, Attention=None, figsize=15, vertical=False,
                    SlidePath='/mnt/e/EWHA/Pancreatic/svs/HE/', filename=None, background_gray=False):
    
    Slide = openslide.open_slide( SlidePath + Barcode)
    TestWSI = np.array(Slide.get_thumbnail(np.array(Slide.dimensions)/HeatmapDownsample)) # 56 down_sample
    if Attention is None:
        PatchMap = np.zeros( np.append(np.array(TestWSI.shape[:2])/(patch_size/HeatmapDownsample)+1, 3).astype(int) )
        print("PatchMap.shape:", PatchMap.shape)
        umap_model = umap.UMAP()
        embedding = umap_model.fit_transform(Feature)
        embedding[:, 0] = minMax(embedding[:, 0])
        embedding[:, 1] = minMax(embedding[:, 1])
        for idx, (Xpos, Ypos) in Coord.iterrows():
            Xpos, Ypos = int(Xpos/patch_size), int(Ypos/patch_size)
            PatchMap[Ypos, Xpos, ] = 1, embedding[idx, 0], embedding[idx, 1]
        PatchMap = cv2.resize(PatchMap, TestWSI.shape[:2][::-1])
    else:
        PatchMap = np.zeros( np.append(np.array(TestWSI.shape[:2])/(patch_size/HeatmapDownsample)+1, 3).astype(int) )
        umap_model = umap.UMAP()
        for idx, (Xpos, Ypos) in Coord.iterrows():
            Xpos, Ypos = int(Xpos/patch_size), int(Ypos/patch_size)
            PatchMap[Ypos, Xpos, ] = 1, Attention[idx, 0], Attention[idx, 1]
        PatchMap = cv2.resize(PatchMap, TestWSI.shape[:2][::-1])

    nImage = PatchMap.shape[2]
    if vertical:
        fig, axes = plt.subplots(nrows=nImage, ncols=1, figsize=(figsize*2, figsize*nImage), constrained_layout=True)
    else:
        fig, axes = plt.subplots(nrows=1, ncols=nImage, figsize=(figsize*nImage, figsize*2), constrained_layout=True)
    axes[0].imshow(TestWSI)
    if background_gray:
        axes[0].imshow(PatchMap[:,:,0], alpha=.2, cmap=cmap_patch)
    for imgIDX in range(1, nImage):
        axes[imgIDX].imshow(TestWSI)
        axes[imgIDX].imshow(PatchMap[:,:,imgIDX], alpha=.5, vmax=1, vmin=0, cmap=cmap_jet)
    if filename:
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        fig.savefig(filename, dpi=300)
        plt.close(fig)
def minMax(x): return (x-x.min())/(x.max()-x.min())

def neighbors(p):
    # 상하좌우와 대각선에 인접한 점들을 반환
    x, y = p
    
    return [(x-1, y), (x+1, y), (x, y-1), (x, y+1), 
            (x-1, y-1), (x+1, y-1), (x-1, y+1), (x+1, y+1)]
def dfs(p, visited, coords_set):
    stack = [p]
    cluster = []

    while stack:
        point = stack.pop()
        if point not in visited:
            visited.add(point)
            cluster.append(point)
            for neighbor in neighbors(point):
                if neighbor in coords_set:
                    stack.append(neighbor)
    return cluster
def filter_small_clusters(coords, threshold=24):
    coords_set = set(map(tuple, coords))
    coords_dict = {tuple(val): idx for idx, val in enumerate(coords)}
    visited = set()
    remaining_coords = []
    remaining_indices = []

    for p in coords_set:
        if p not in visited:
            cluster = dfs(p, visited, coords_set)
            if len(cluster) > threshold:
                remaining_coords.extend(cluster)
                remaining_indices.extend([coords_dict[c] for c in cluster])
    return np.array(remaining_coords), np.array(remaining_indices)


def main(args):
    if args.dino_patch_size == 8:
        key = "DINO_p8"
    elif args.dino_patch_size == 16:
        key = "DINO_p16"
    else:
        print("patch_size should be 8 or 16")
        exit

    svs_files = glob.glob(f"{args.WSI_directory}*.svs")
    tif_files = glob.glob(f"{args.WSI_directory}*.tif")
    Slide_filenames = svs_files + tif_files

    os.makedirs(args.BaseSavePath, exist_ok=True)
    Path_Coord   = os.path.join(args.BaseSavePath, f"Coord_{args.backbone}")
    Path_Feature = os.path.join(args.BaseSavePath, f"Feature_{args.backbone}")
    Path_Heatmap = os.path.join(args.BaseSavePath, f"Heatmap_{args.backbone}")

    os.makedirs(Path_Coord,   exist_ok=True)
    os.makedirs(Path_Feature, exist_ok=True)
    os.makedirs(Path_Heatmap, exist_ok=True)

    # [idx for idx, i in enumerate(Slide_filenames) if os.path.basename(i) == 'GI117_13011_HE.svs'] GI117_13011_HE.svs
    curBarcode = Slide_filenames[args.slide_idx].split('/')[-1]
    if os.path.exists(f'{Path_Heatmap}/{curBarcode}.png'):
        print(f'{Path_Heatmap}/{curBarcode}.png already exists')
        exit()


    ###########################################################################
    ## 1. Model Setting #######################################################
    ###########################################################################
    transform = transforms.Compose(
        [
            #transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    
    if args.backbone == 'UNI':
        local_dir = "./vit_large_patch16_224.dinov2.uni_mass100k/"
        os.makedirs(local_dir, exist_ok=True)  # create directory if it does not exist
        #hf_hub_download("MahmoodLab/UNI", filename="pytorch_model.bin", local_dir=local_dir, force_download=False)
        MODEL = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
        MODEL.load_state_dict(torch.load(os.path.join(local_dir, "pytorch_model.bin"), map_location="cpu"), strict=True)
        MODEL.cuda()
        MODEL.eval()
        print("UNI model Load completed")
    elif args.backbone == 'DINO':
        MODEL = vit_small(pretrained=True, progress=False, key="DINO_p8", patch_size=args.dino_patch_size)
        MODEL = MODEL.cuda()
        MODEL.eval()
        print("LUNIT-DINO model Load completed")
    elif args.backbone == 'GP':
        MODEL = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        MODEL.cuda()
        MODEL.eval()
    
    ###########################################################################
    ## 2. Load Data ###########################################################
    ###########################################################################

    PatchGenerator = PatchDataset(Slide_filenames[args.slide_idx], patch_size=args.patch_size,
                                  stride=args.stride, downsample=args.downsample, transform=transform)
    PatchDataLoader = torch.utils.data.DataLoader(PatchGenerator, batch_size=args.batch_size, num_workers=0, pin_memory=False, shuffle=False)

    ###########################################################################
    ## 3. Feature Extraction ##################################################
    ###########################################################################
    Features, Outputs, Xcoords, Ycoords = [], [], [], []
    with torch.no_grad():  # no_grad 블록 추가
        for Tensors, Position in tqdm(PatchDataLoader):
            Feature = MODEL(Tensors.float().cuda())  # Tensors도 GPU로 옮기기
            Features.append(Feature.cpu().detach().numpy())  # GPU에서 CPU로 데이터 이동
            Xcoords.append(Position[0].tolist())
            Ycoords.append(Position[1].tolist())

    Features = np.vstack(Features)
    Xpos = np.concatenate(Xcoords)
    Ypos = np.concatenate(Ycoords)
    Coords = pd.DataFrame({'Xpos': Xpos, 'Ypos': Ypos})

    ###########################################################################
    ## 4. island tissue filter ################################################
    ###########################################################################
    Coords_idx = (np.array(Coords)/args.patch_size).astype('int')

    threshold = 50
    if len(Features) < 3000:
        threshold = 0
    filtered_coords, indices = filter_small_clusters(Coords_idx, threshold=threshold)
    Features_filtered = Features[indices,]
    Coords_filtered = Coords.loc[indices,].reset_index(drop=True)

    ###########################################################################
    ## 5. Feature, Coords, Heatmap Visualization ##############################
    ###########################################################################
    cmap = plt.cm.jet
    cmap_jet = cmap(np.arange(cmap.N))
    cmap_jet[:1, -1] = 0  # 첫번째 색상을 투명하게 설정
    cmap_jet = ListedColormap(cmap_jet)

    cmap_patch = np.array([[0, 0, 0, 1], [0, 0, 0, 0]])  # Black & Transparent
    cmap_patch = ListedColormap(cmap_patch)    

    with open(f'{Path_Feature}/{curBarcode}.pickle', 'wb') as f:
        pickle.dump(Features_filtered, f, pickle.HIGHEST_PROTOCOL)
    with open(f'{Path_Coord}/{curBarcode}.pickle', 'wb') as f:
        pickle.dump(Coords_filtered, f, pickle.HIGHEST_PROTOCOL)
    DrawUmapHeatmap(curBarcode, Features_filtered, Coords_filtered, patch_size=args.patch_size, HeatmapDownsample=224,
                    figsize=7, vertical=True, SlidePath=args.WSI_directory, filename=f'{Path_Heatmap}/{curBarcode}.png')


if __name__ == "__main__":
    # argparse를 사용하여 인수 정의
    # sys.argv = ['script_name', '--slide_idx', '1']  # Replace '1' with the desired slide index value
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("--patch_size", type=int, default=448, help="patch size")
    parser.add_argument("--dino_patch_size", type=int, default=8, help="patch size")
    parser.add_argument("--BaseSavePath", type=str, default="./Features_AllTypes/", help="base save path")
    parser.add_argument("--stride", type=int, default=448, help="stride")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--downsample", type=int, default=2, help="downsample factor")
    parser.add_argument("--WSI_directory", type=str, default="/mnt/f/LIHC_CHOL_LUAD_COAD_ESCA/svs/", help="downsample factor")
    parser.add_argument("--backbone", type=str, default='UNI', help="DINO, UNI and GP")
    parser.add_argument("--slide_idx", type=int, required=True, help="Index of the slide to process")
    
    args = parser.parse_args()
    main(args)
