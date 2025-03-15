from multiprocessing import Pool
import subprocess

def run_script_gp(slide_idx):
    subprocess.run(["python", "01.UNI_PatchFeatureExtractor_HE_20.py", "--slide_idx", str(slide_idx), "--backbone", "GP" ])

def run_script_dino(slide_idx):
    subprocess.run(["python", "01.UNI_PatchFeatureExtractor_HE_20.py", "--slide_idx", str(slide_idx), "--backbone", "DINO" ])


def run_script_AllInOne_TCGA(slide_idx):
    subprocess.run(["python", "01.UNI_PatchFeatureExtractor_TCGA_AllInOne.py", "--slide_idx", str(slide_idx)])

if __name__ == "__main__":
    with Pool(3) as p:
        p.map(run_script_AllInOne_TCGA, range(209))