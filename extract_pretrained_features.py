import os
import glob
import time
import shutil
import pickle
import timm
import clip
import open_clip
import sklearn
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
import open_clip
from torchvision import transforms as pth_transforms
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import warnings
warnings.simplefilter("ignore", sklearn.exceptions.DataConversionWarning)

#%% helper functions

def load_timm_model(model_name='convnext_xlarge_in22k', device='cpu'):

    pretrained_model = timm.create_model(model_name, pretrained=True, num_classes=0).eval().to(device)
    model_config_dict = resolve_data_config({}, model=pretrained_model)
    model_preprocess = create_transform(**model_config_dict)

    return pretrained_model, model_preprocess

def load_dino_model(model_name='dino_vitb8', device='cpu'):

    model_preprocess = pth_transforms.Compose([
            pth_transforms.Resize(256, interpolation=3),
            pth_transforms.CenterCrop(224),
            pth_transforms.ToTensor(),
            pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    pretrained_model = torch.hub.load('facebookresearch/dino:main', model_name).to(device)

    return pretrained_model, model_preprocess


def load_openclip_model(model_name, device="cpu"):
    if model_name == "OpenCLIP_ViT-H-14-378-quickgelu":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14-378-quickgelu', pretrained='dfn5b')
    elif model_name == "OpenCLIP_ViT-bigG-14-CLIPA-336":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14-CLIPA-336', pretrained='datacomp1b')
    elif model_name == "OpenCLIP_ViT-SO400M-14-SigLIP-384":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-SO400M-14-SigLIP', pretrained='webli')
    elif model_name == "OpenCLIP_ViT-G-14":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-bigG-14', pretrained='laion2b_s39b_b160k')
    elif model_name == "OpenCLIP_ConvNext-XXLarge":
        model, _, preprocess = open_clip.create_model_and_transforms('convnext_xxlarge', pretrained='laion2b_s34b_b82k_augreg_soup')
    elif model_name == "OpenCLIP_ViT-H-14":
        model, _, preprocess = open_clip.create_model_and_transforms('ViT-H-14', pretrained='laion2b_s32b_b79k')
    else:
        model, _, preprocess = open_clip.create_model_and_transforms(model_name)
    
    model = model.to(device)
    model.eval()
    
    return model, preprocess

def extract_pretrained_features(base_image_folder, model_to_use='CLIP_ViTL_14@336'):

    images_folder = os.path.join(base_image_folder, 'images')
    transfer_images = False

    # First, determine if we need to transfer images and which images to transfer if we do
    if not os.path.exists(images_folder):
        base_image_files = []
        for image_file_ending in ['*.jpg', '*.png']:
            base_image_files.extend(glob.glob(os.path.join(base_image_folder, image_file_ending)))
        if base_image_files:
            transfer_images = True

    # Now, handle the images based on whether we need to transfer
    if transfer_images:
        os.makedirs(images_folder, exist_ok=True)
        for src_image_filename in base_image_files:
            shutil.move(src_image_filename, images_folder)
        all_image_filenames = [os.path.join(images_folder, os.path.basename(f)) for f in base_image_files]
        print('Images were transferred to the images folder.')
    else:
        if os.path.exists(images_folder):
            all_image_filenames = glob.glob(os.path.join(images_folder, '*.*'))
            print('Images were already in the correct location.')
        else:
            print('No images found in the base folder or in an "images" subfolder.')
            return

    if len(all_image_filenames) == 0:
        print('No images found to process.')
        return

    # Create features folder if it doesn't exist
    features_folder = os.path.join(base_image_folder, 'pretrained_features')
    os.makedirs(features_folder, exist_ok=True)

    # load requested model
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f'loading model "{model_to_use}"...')

    if   model_to_use == 'CLIP_ViTL_14@336':
        pretrained_model, model_preprocess = clip.load("ViT-L/14@336px", device=device)
    elif model_to_use == 'CLIP_ViTL_14':
        pretrained_model, model_preprocess = clip.load("ViT-L/14", device=device)
    elif model_to_use == 'CLIP_ViTB_16':
        pretrained_model, model_preprocess = clip.load("ViT-B/16", device=device)
    elif model_to_use == 'CLIP_ViTB_32':
        pretrained_model, model_preprocess = clip.load("ViT-B/32", device=device)
    elif model_to_use == 'CLIP_ResNet50x64':
        pretrained_model, model_preprocess = clip.load("RN50x64", device=device)
    elif model_to_use == 'CLIP_ResNet50x16':
        pretrained_model, model_preprocess = clip.load("RN50x16", device=device)
    elif model_to_use == 'CLIP_ResNet50x4':
        pretrained_model, model_preprocess = clip.load("RN50x4", device=device)
    elif model_to_use == 'CLIP_ResNet50x1':
        pretrained_model, model_preprocess = clip.load("RN50", device=device)
    elif model_to_use == 'CLIP_ResNet101':
        pretrained_model, model_preprocess = clip.load("RN101", device=device)

    elif model_to_use == 'DINO_ResNet50':
        pretrained_model, model_preprocess = load_dino_model("dino_resnet50", device=device)
    elif model_to_use == 'DINO_ViTS_8':
        pretrained_model, model_preprocess = load_dino_model("dino_vits8", device=device)
    elif model_to_use == 'DINO_ViTB_8':
        pretrained_model, model_preprocess = load_dino_model("dino_vitb8", device=device)

    elif model_to_use == 'ConvNext_XL_Imagenet21k':
        pretrained_model, model_preprocess = load_timm_model(model_name='convnext_xlarge_in22k', device=device)
    elif model_to_use == 'ConvNext_XL_384_Imagenet21k_ft_1k':
        pretrained_model, model_preprocess = load_timm_model(model_name='convnext_xlarge_384_in22ft1k', device=device)
    elif model_to_use == 'ConvNext_L_Imagenet21k':
        pretrained_model, model_preprocess = load_timm_model(model_name='convnext_large_in22k', device=device)
    elif model_to_use == 'ConvNext_L_384_Imagenet21k_ft_1k':
        pretrained_model, model_preprocess = load_timm_model(model_name='convnext_large_384_in22ft1k', device=device)

    elif model_to_use == 'EffNet_L2_NS_475':
        pretrained_model, model_preprocess = load_timm_model(model_name='tf_efficientnet_l2_ns_475', device=device)
    elif model_to_use == 'EffNet_B7_NS_600':
        pretrained_model, model_preprocess = load_timm_model(model_name='tf_efficientnet_b7_ns', device=device)
    elif model_to_use == 'EffNetV2_L_480_Imagenet21k_ft_1k':
        pretrained_model, model_preprocess = load_timm_model(model_name='tf_efficientnetv2_l_in21ft1k', device=device)
    elif model_to_use == 'EffNetV2_S_384_Imagenet21k_ft_1k':
        pretrained_model, model_preprocess = load_timm_model(model_name='tf_efficientnetv2_s_in21ft1k', device=device)

    elif model_to_use == 'BEiT_L_16_512':
        pretrained_model, model_preprocess = load_timm_model(model_name='beit_large_patch16_512', device=device)
    elif model_to_use == 'BEiT_L_16_384':
        pretrained_model, model_preprocess = load_timm_model(model_name='beit_large_patch16_384', device=device)
    elif model_to_use == 'BEiT_L_16_224':
        pretrained_model, model_preprocess = load_timm_model(model_name='beit_large_patch16_224', device=device)

    elif model_to_use == 'DeiT3_L_16_384_Imagenet21k_ft_1k':
        pretrained_model, model_preprocess = load_timm_model(model_name='deit3_large_patch16_384_in21ft1k', device=device)
    elif model_to_use == 'DeiT3_H_14_224_Imagenet21k_ft_1k':
        pretrained_model, model_preprocess = load_timm_model(model_name='deit3_huge_patch14_224_in21ft1k', device=device)
    elif model_to_use == 'DeiT3_L_16_224_Imagenet21k_ft_1k':
        pretrained_model, model_preprocess = load_timm_model(model_name='deit3_large_patch16_224_in21ft1k', device=device)
    
    elif model_to_use == 'OpenCLIP_ViT-bigG-14-CLIPA-336':
        pretrained_model, model_preprocess = load_openclip_model("OpenCLIP_ViT-bigG-14-CLIPA-336", device=device)
    elif model_to_use == 'OpenCLIP_ViT-H-14-378-quickgelu':
        pretrained_model, model_preprocess = load_openclip_model("OpenCLIP_ViT-H-14-378-quickgelu", device=device)
    elif model_to_use == 'OpenCLIP_ViT-SO400M-14-SigLIP-384':
        pretrained_model, model_preprocess = load_openclip_model("OpenCLIP_ViT-SO400M-14-SigLIP-384", device=device)
    elif model_to_use == 'OpenCLIP_ViT-G-14':
        pretrained_model, model_preprocess = load_openclip_model("OpenCLIP_ViT-G-14", device=device)
    elif model_to_use == 'OpenCLIP_ConvNext-XXLarge':
        pretrained_model, model_preprocess = load_openclip_model("OpenCLIP_ConvNext-XXLarge", device=device)
    elif model_to_use == 'OpenCLIP_ViT-H-14':
        pretrained_model, model_preprocess = load_openclip_model("OpenCLIP_ViT-H-14", device=device)
    else:
        print('unrecognized modelname, not calculated any features!')
        return

    print(f'"{model_to_use}" model loaded')
    print(f'Calculating {len(all_image_filenames)} features of model "{model_to_use}"...')

    start_time = time.time()
    # Go over all images and append features to features dict
    for curr_image_filename in tqdm(all_image_filenames, desc=f'Extracting "{model_to_use}" features', unit="image"):
        curr_sample_name = os.path.splitext(os.path.basename(curr_image_filename))[0]
        curr_features_dict_filename = os.path.join(features_folder, curr_sample_name + '.pickle')

        # Check if features_dict file exists, if it doesn't, create one
        if os.path.isfile(curr_features_dict_filename):
            with open(curr_features_dict_filename, "rb") as f:
                curr_features_dict = pickle.load(f)
        else:
            curr_features_dict = {}

        # If the requested features were already calculated for this sample, skip it
        if model_to_use in curr_features_dict.keys():
            continue

        # Extract the features
        curr_image_PIL = Image.open(curr_image_filename).convert("RGB")

        with torch.no_grad():
            if 'CLIP' in model_to_use:
                curr_pretrained_features = pretrained_model.encode_image(model_preprocess(curr_image_PIL).unsqueeze(0).to(device))
            elif 'OpenCLIP' in model_to_use:
                curr_pretrained_features = pretrained_model.encode_image(model_preprocess(curr_image_PIL).unsqueeze(0).to(device))
            elif 'DINO' in model_to_use:
                curr_pretrained_features = pretrained_model(model_preprocess(curr_image_PIL).unsqueeze(0).to(device))
            elif 'ConvNext' in model_to_use:
                curr_pretrained_features = pretrained_model(model_preprocess(curr_image_PIL).unsqueeze(0).to(device))
            elif 'EffNet' in model_to_use:
                curr_pretrained_features = pretrained_model(model_preprocess(curr_image_PIL).unsqueeze(0).to(device))
            elif 'BEiT' in model_to_use:
                curr_pretrained_features = pretrained_model(model_preprocess(curr_image_PIL).unsqueeze(0).to(device))
            elif 'DeiT' in model_to_use:
                curr_pretrained_features = pretrained_model(model_preprocess(curr_image_PIL).unsqueeze(0).to(device))

        curr_features_dict[model_to_use] = curr_pretrained_features.detach().cpu().numpy()

        # Save the dictionary
        with open(curr_features_dict_filename, "wb") as f:
            pickle.dump(curr_features_dict, f)

    total_duration_min = (time.time() - start_time) / 60
    print(f'Extracted "{model_to_use}" features from {len(all_image_filenames)} images. Total time: {total_duration_min:.2f} minutes')

    return

def collect_pretrained_features_from_folder(base_image_folder, model_name, normalize_features=True, ignore_DALLE=True):
    features_folder = os.path.join(base_image_folder, 'pretrained_features')
    all_feature_dict_filenames = glob.glob(os.path.join(features_folder, '*.pickle'))
    all_image_filenames = glob.glob(os.path.join(base_image_folder, 'images', '*.*'))

    if ignore_DALLE:
        all_feature_dict_filenames = [x for x in all_feature_dict_filenames if 'DALLE3' not in x]
        all_image_filenames = [x for x in all_image_filenames if 'DALLE3' not in x]

    features_list = []
    image_filename_map = {}

    for curr_image_filename in all_image_filenames:
        curr_sample_name = os.path.splitext(os.path.basename(curr_image_filename))[0]
        curr_features_dict_filename = os.path.join(features_folder, f"{curr_sample_name}.pickle")
        
        if not os.path.exists(curr_features_dict_filename):
            continue  # Skip images without corresponding feature dict
        
        with open(curr_features_dict_filename, "rb") as f:
            curr_features_dict = pickle.load(f)
        
        if model_name not in curr_features_dict:
            continue
        
        features_list.append(curr_features_dict[model_name])
        image_filename_map[len(features_list)-1] = curr_image_filename

        if len(features_list) % 5000 == 0:
            print(f"Processed {len(features_list)} images...")

    if not features_list:
        raise ValueError("No features were collected. Please check your directories and files.")

    pretrained_image_features = np.vstack(features_list)

    # Normalize features if requested
    if normalize_features:
        pretrained_image_features /= np.linalg.norm(pretrained_image_features, axis=1)[:, np.newaxis]

    return pretrained_image_features, image_filename_map

def extract_and_collect_pretrained_features(images_base_folder, models_to_use=['CLIP_ViTL_14@336','CLIP_ResNet50x64'], nromalize_features=True):
    # this function will extract the features of all models in "models_to_use", collect the  and concatenate them

    # extracting features
    for model_to_use in models_to_use:
        extract_pretrained_features(images_base_folder, model_to_use=model_to_use)

    # collecting features
    features_list = []
    image_filename_map_list = []
    for requested_features_model in models_to_use:
        image_features, image_filename_map = collect_pretrained_features_from_folder(images_base_folder, requested_features_model, nromalize_features=nromalize_features)
        features_list.append(image_features)
        image_filename_map_list.append(image_filename_map)

    # make sure the maps are identical
    try:
        for k in range(len(image_filename_map_list) - 1):
            for key in image_filename_map_list[k].keys():
                assert image_filename_map_list[k][key] == image_filename_map_list[k + 1][key]
    except:
        print('the maps are not identical. quitting')
        return

    # concatenate the features
    combined_image_features = np.concatenate(features_list, axis=1)

    return combined_image_features, image_filename_map_list[0]

def delete_near_duplicates(base_image_folder, models_to_use=['CLIP_ViTL_14@336','CLIP_ResNet50x64'], similarity_threshold=0.99, minibatch_size=10_000):
    # this function does not assume "proper" folder stucture, but will create it and calculate features if necessary

    features_folder = os.path.join(base_image_folder, 'pretrained_features')

    # collect the requested features to calculate near duplication based on
    image_features, image_filename_map = extract_and_collect_pretrained_features(base_image_folder, models_to_use=models_to_use, nromalize_features=True)
    similarity_threshold = len(models_to_use) * similarity_threshold

    total_num_samples = image_features.shape[0]
    num_batches = np.ceil(total_num_samples / minibatch_size).astype(int)

    feature_inds_to_drop = []

    end_row_ind = 0
    for batch_ind in range(num_batches):
        start_row_ind = end_row_ind
        end_row_ind = min(start_row_ind + minibatch_size, total_num_samples)
        image_feature_curr_batch = image_features[start_row_ind:end_row_ind]
        curr_minibatch_size = image_feature_curr_batch.shape[0]

        similarity_curr_batch_to_all = np.dot(image_feature_curr_batch, image_features.T).astype(np.float32)
        similarity_curr_batch_to_all[np.arange(curr_minibatch_size), np.arange(start_row_ind, end_row_ind)] = 0
        similarity_curr_batch_to_all = similarity_curr_batch_to_all > similarity_threshold

        # zero out all removals from previous batches
        if len(feature_inds_to_drop) > 0:
            similarity_curr_batch_to_all[:,np.array(feature_inds_to_drop)] = 0

        # go over the self similarity matrix rows and determine which indices should be removed
        for curr_batch_row_ind in range(curr_minibatch_size):
            if similarity_curr_batch_to_all[curr_batch_row_ind,:].sum() > 0:
                full_features_row = start_row_ind + curr_batch_row_ind
                feature_inds_to_drop.append(full_features_row)
                # zero out the column of the removed duplicate (so that it's twins won't be removed as well)
                similarity_curr_batch_to_all[:,full_features_row] = 0

    num_to_remove = len(feature_inds_to_drop)
    message_string = 'from the folder "%s" (contains %d images) \nthere will be removed %d near-duplicates (%.1f%s of images)'
    print('----------------------------------------')
    print(message_string %(base_image_folder, total_num_samples, num_to_remove, 100 * (num_to_remove / total_num_samples), '%'))
    print('----------------------------------------')

    # remove the files
    for k in feature_inds_to_drop:
        curr_image_filename = image_filename_map[k]
        curr_sample_name = curr_image_filename.split('/')[-1].split('.')[0]
        curr_features_dict_filename = os.path.join(features_folder, curr_sample_name + '.pickle')

        os.remove(curr_image_filename)
        os.remove(curr_features_dict_filename)


def find_nearest_neighbors(folder_A, folder_B, model_name, k=5):
    # Collect features from both folders
    features_A, filenames_A = collect_pretrained_features_from_folder(folder_A, model_name)
    features_B, filenames_B = collect_pretrained_features_from_folder(folder_B, model_name)

    print(f"Folder A features shape: {features_A.shape}")
    print(f"Folder B features shape: {features_B.shape}")

    nearest_neighbors = []
    
    # Iterate over each image index in folder A
    for i in tqdm(range(len(features_A)), desc="Finding nearest neighbors"):

        # Find the top k nearest neighbors in folder B
        feature_A = features_A[i:i+1]
        similarities = np.dot(feature_A, features_B.T).flatten()
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        
        # Get filenames and similarities of nearest neighbors
        neighbor_filenames = [filenames_B[idx] for idx in top_k_indices]
        neighbor_similarities = [similarities[idx] for idx in top_k_indices]
        
        nearest_neighbors.append({
            'source_image': filenames_A[i],
            'neighbors': list(zip(neighbor_filenames, neighbor_similarities))
        })

    return nearest_neighbors
