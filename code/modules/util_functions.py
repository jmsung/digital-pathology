from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')  # Use a backend that does not require a display
from matplotlib.colors import ListedColormap
import torch
import numpy as np
import openslide

def minMax(x): return (x-x.min())/(x.max()-x.min())
def DrawUmapHeatmap(Barcode, Feature, Coord, cmap_patch, cmap_jet, HeatmapDownsample = 14,
                    Attention=None, figsize=15, vertical=False, patch_size=224,
                    SlidePath='/mnt/e/JNUH/STAD_WSI/Gastric_Ca_ImmunoTx/', filename=None, background_gray=False, alpha=0.5,
                    ): # patch_size 224 for NCC, but 448 for TCGA
    
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
        axes[imgIDX].imshow(PatchMap[:,:,imgIDX], alpha=alpha, vmax=1, vmin=0, cmap=cmap_jet)
    if filename:
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)
        fig.savefig(filename, dpi=300)
        plt.close(fig)


def plotSurvival(data):
    kmf = KaplanMeierFitter()
    colors = {True: 'blue', False: 'green'}
    # PredClass에 따라 그룹화하여 생존 곡선 그리기
    for pred_class in data['PredClass'].unique():
        mask = data['PredClass'] == pred_class
        kmf.fit(durations=data['times'][mask], event_observed=data['events'][mask], label=f'PredClass {pred_class}')
        kmf.plot_survival_function(ci_show=True, color=colors[pred_class])

    plt.title('Survival Analysis: Kaplan-Meier Estimates by PredClass')
    plt.xlabel('Time')
    plt.ylabel('Survival Probability')
    plt.legend()


def plotSurvival_two(data_train, data_valid):
    kmf = KaplanMeierFitter()
    colors = {True: 'blue', False: 'green'}
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1행 2열 subplot 생성
    
    # data_train kaplan meier
    for pred_class in data_train['PredClass'].unique():
        mask = data_train['PredClass'] == pred_class
        kmf.fit(durations=data_train['times'][mask], event_observed=data_train['events'][mask], label=f'PredClass {pred_class}')
        kmf.plot_survival_function(ax=axes[0], ci_show=True, color=colors[pred_class], show_censors=True)  # 첫 번째 subplot

    axes[0].set_title('Train Dataset Survival Analysis')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Survival Probability')
    axes[0].legend()
    axes[0].text(0.05, 0.05, f'p-value: {logRankTest(data_train):.4f}', transform=axes[0].transAxes)

    # data_valid kaplan meier
    for pred_class in data_valid['PredClass'].unique():
        mask = data_valid['PredClass'] == pred_class
        kmf.fit(durations=data_valid['times'][mask], event_observed=data_valid['events'][mask], label=f'PredClass {pred_class}')
        kmf.plot_survival_function(ax=axes[1], ci_show=True, color=colors[pred_class], show_censors=True)  # 두 번째 subplot

    axes[1].set_title('Validation Dataset Survival Analysis')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Survival Probability')
    axes[1].legend()
    axes[1].text(0.05, 0.05, f'p-value: {logRankTest(data_valid):.4f}', transform=axes[1].transAxes)

    plt.tight_layout()
    
    return logRankTest(data_valid)

def plotSurvival_three(data_train, data_valid, data_test, fig_size=(18,5), filename=None):
    kmf = KaplanMeierFitter()
    colors = {True: 'blue', False: 'green'}
    
    fig, axes = plt.subplots(1, 3, figsize=fig_size)    
    ############# data_train
    for pred_class in data_train['PredClass'].unique():
        mask = data_train['PredClass'] == pred_class
        kmf.fit(durations=data_train['times'][mask], event_observed=data_train['events'][mask], label=f'PredClass {pred_class}')
        kmf.plot_survival_function(ax=axes[0], ci_show=True, color=colors[pred_class], show_censors=True)  # 첫 번째 subplot

    axes[0].set_title('Train Dataset Survival Analysis')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Survival Probability')
    axes[0].legend()
    axes[0].text(0.05, 0.05, f'p-value: {logRankTest(data_train):.4f}', transform=axes[0].transAxes)

    ############# data_valid
    for pred_class in data_valid['PredClass'].unique():
        mask = data_valid['PredClass'] == pred_class
        kmf.fit(durations=data_valid['times'][mask], event_observed=data_valid['events'][mask], label=f'PredClass {pred_class}')
        kmf.plot_survival_function(ax=axes[1], ci_show=True, color=colors[pred_class], show_censors=True)  # 두 번째 subplot

    axes[1].set_title('Validation Dataset Survival Analysis')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Survival Probability')
    axes[1].legend()
    axes[1].text(0.05, 0.05, f'p-value: {logRankTest(data_valid):.4f}', transform=axes[1].transAxes)

    ############# data_NCC
    for pred_class in data_test['PredClass'].unique():
        mask = data_test['PredClass'] == pred_class
        kmf.fit(durations=data_test['times'][mask], event_observed=data_test['events'][mask], label=f'PredClass {pred_class}')
        kmf.plot_survival_function(ax=axes[2], ci_show=True, color=colors[pred_class], show_censors=True)  # 두 번째 subplot

    axes[2].set_title('Test Dataset Survival Analysis')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Survival Probability')
    axes[2].legend()
    axes[2].text(0.05, 0.05, f'p-value: {logRankTest(data_test):.4f}', transform=axes[2].transAxes)
    plt.tight_layout()  # subplot 간격 조정
    if filename:
        plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    return logRankTest(data_valid), logRankTest(data_test)
    
def logRankTest(data):
    Time_A  = data.loc[data['PredClass'] == True , 'times']
    Event_A = data.loc[data['PredClass'] == True , 'events']
    Time_B  = data.loc[data['PredClass'] == False, 'times']
    Event_B = data.loc[data['PredClass'] == False, 'events']
    
    results = logrank_test(Time_A, Time_B, event_observed_A=Event_A, event_observed_B=Event_B)
    return results.p_value

def run_one_sample(Feature, model, conf, device):
    image_patches = torch.from_numpy(Feature).unsqueeze(0).float().to(device)
    sub_preds, slide_preds, attn = model(image_patches, use_attention_mask=True)
    #loss_similarity_weights = criterion(sub_preds, labels.repeat_interleave(conf.n_token))
    loss_similarity_distance = torch.tensor(0).float().to(device)
    attn = torch.softmax(attn, dim=-1)
    for i in range(conf.n_token):
        for j in range(i + 1, conf.n_token):
            loss_similarity_distance += torch.cosine_similarity(attn[:, i], attn[:, j], dim=-1).mean() / (
                        conf.n_token * (conf.n_token - 1) / 2)
    #self_loss = loss_similarity_distance + loss_similarity_weights
    return loss_similarity_distance, slide_preds

def featureRandomSelection(Feature, random_ratio=(0.8, 1.0)):
    random_ratio = np.random.uniform(random_ratio[0], random_ratio[1])
    num_rows = Feature.shape[0]
    num_samples = int(num_rows * random_ratio)
    random_indices = np.random.choice(num_rows, size=num_samples, replace=False)
    return Feature[random_indices]