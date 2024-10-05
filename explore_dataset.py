#%% Imports 

import os
import glob
import torch
import open_clip
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from extract_pretrained_features import extract_pretrained_features, load_openclip_model
from extract_pretrained_features import collect_pretrained_features_from_folder

#%% Helper functions

def plot_model_distribution(df):
    model_counts = df['model_used'].value_counts().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(model_counts.index, model_counts.values)
    ax.set_title('Distribution of Images Across Models')
    ax.set_xlabel('Model')
    ax.set_ylabel('Number of Images')
    plt.xticks(rotation=45, ha='right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Add value labels on top of each bar
    for i, v in enumerate(model_counts.values):
        ax.text(i, v, str(v), ha='center', va='bottom')
    
    return fig

def plot_prompt_length_distribution(df):
    # Calculate prompt lengths
    df['prompt_chars'] = df['text_prompt'].str.len()
    df['prompt_words'] = df['text_prompt'].str.split().str.len()

    # Set up the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

    # Colors for each model
    models = df['model_used'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(models)))

    # Plot character length distribution
    for model, color in zip(models, colors):
        model_data = df[df['model_used'] == model]['prompt_chars']
        ax1.hist(model_data, bins=50, alpha=0.5, label=model, color=color)
    
    ax1.set_title('Prompt Length in Characters', fontsize=15)
    ax1.set_xlabel('Number of Characters', fontsize=13)
    ax1.set_ylabel('Frequency', fontsize=13)
    # ax1.set_yscale('log')
    ax1.legend(fontsize=14)

    # Plot word length distribution
    for model, color in zip(models, colors):
        model_data = df[df['model_used'] == model]['prompt_words']
        ax2.hist(model_data, bins=20, alpha=0.5, label=model, color=color)
    
    ax2.set_title('Prompt Length in Words', fontsize=15)
    ax2.set_xlabel('Number of Words', fontsize=13)
    ax2.set_ylabel('Frequency', fontsize=13)
    # ax2.set_yscale('log')
    ax2.legend(fontsize=14)

    plt.tight_layout()
    return fig

def format_prompt(prompt, max_width=85, min_width=55):
    words = prompt.split()
    lines = []
    current_line = ""

    for word in words:
        if len(current_line) + len(word) + 1 > max_width:
            lines.append(current_line)
            current_line = word
        else:
            if current_line:
                current_line += " " + word
            else:
                current_line = word

            if len(current_line) >= min_width and (current_line.endswith(',') or current_line.endswith('.')):
                lines.append(current_line) 
                current_line = ""

    if current_line:
        lines.append(current_line)

    return '\n'.join(lines)

def remove_borders(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

def display_single_random_image(df, image_folder, model_to_use=None):
    if model_to_use is not None:
        df_to_use = df[df['model_used'] == model_to_use]
    else:
        df_to_use = df
    
    random_row = df_to_use.sample(n=1).iloc[0]
    image_path = os.path.join(image_folder, random_row['image_filename'])
    
    img = Image.open(image_path)
    
    fig, ax = plt.subplots(figsize=(14, 11))
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.2)
    ax.imshow(img)
    
    title = f"Model: {random_row['model_used']}, filename = '{random_row['image_filename']}'"
    ax.set_title(title, fontsize=14)
    
    formatted_prompt = format_prompt(random_row['text_prompt'])
    ax.set_xlabel(formatted_prompt, fontsize=11)
    
    return fig

def display_single_model_images(df, image_folder, model_to_use=None):
    if model_to_use is not None:
        df_to_use = df[df['model_used'] == model_to_use]
    else:
        df_to_use = df

    if len(df_to_use) < 2:
        raise ValueError("Not enough images to display.")

    random_rows = df_to_use.sample(n=2)

    image_paths = []
    titles = []
    prompts = []
    for _, row in random_rows.iterrows():
        image_path = os.path.join(image_folder, row['image_filename'])
        image_paths.append(image_path)
        title = f"Model: {row['model_used']}\nfilename = '{row['image_filename']}'"
        titles.append(title)
        formatted_prompt = format_prompt(row['text_prompt'], max_width=80, min_width=55)
        prompts.append(formatted_prompt)

    fig, axes = plt.subplots(1, 2, figsize=(20, 13))
    fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.2, wspace=0.05)

    for ax, img_path, title, prompt in zip(axes, image_paths, titles, prompts):
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(title, fontsize=18)
        remove_borders(ax)
        ax.set_xlabel(prompt, fontsize=15, ha='center', va='top')

    return fig

def display_multi_model_images(df, image_folder, num_cols=5, models_names=None, display_prompt=False):
    if models_names is None:
        models_names = df['model_used'].unique().tolist()
    num_rows = len(models_names)
    
    if display_prompt:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 6*num_rows))
        fig.subplots_adjust(hspace=0.3, wspace=0.2, top=0.9)
    else:
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
        fig.subplots_adjust(hspace=0.12, wspace=0.04)
    
    for row, model in enumerate(models_names):
        model_df = df[df['model_used'] == model]
        sample_df_rows = model_df.sample(n=min(num_cols, len(model_df)))
        
        for col, (_, sample_df_row) in enumerate(sample_df_rows.iterrows()):
            ax = axes[row, col] if num_rows > 1 else axes[col]
            
            image_path = os.path.join(image_folder, sample_df_row['image_filename'])
            img = Image.open(image_path)
            
            ax.imshow(img)
            remove_borders(ax)

            ax.set_title(sample_df_row['image_filename'], fontsize=10, wrap=True)
            
            if col == 0:
                ax.set_ylabel(model, fontsize=16)
            
            if display_prompt:
                formatted_prompt = format_prompt(sample_df_row['text_prompt'], max_width=40, min_width=20)
                ax.text(0, -0.1, formatted_prompt, fontsize=5, ha='left', va='top', 
                        wrap=True, transform=ax.transAxes)
    
    return fig

#%%

if __name__ == "__main__":

    #%% Set the style of the plots
    plt.style.use('dark_background')

    # Set the paths
    dataset_path = r"SFHQ_T2I_dataset"

    csv_path = os.path.join(dataset_path, "SFHQ_T2I_dataset.csv")
    image_folder = os.path.join(dataset_path, "images")

    # Load the dataset csv file
    df = pd.read_csv(csv_path)

    # print several randomly selected prompts to screen
    for i in range(10):
        random_row = df.sample(n=1).iloc[0]
        curr_row_prompt = random_row['text_prompt']
        num_chars = len(curr_row_prompt)
        num_words = len(curr_row_prompt.split())
        print('=' * 90)
        print(f'num chars: {num_chars}, num words: {num_words}')
        print('-' * 30)
        print(format_prompt(curr_row_prompt))
    print('=' * 90)

    # Plot the distribution of images across models
    fig_distribution = plot_model_distribution(df)

    # Plot the distribution of prompt lengths
    fig_prompt_lengths = plot_prompt_length_distribution(df)

    # Display two random images for each model
    model_to_use = 'FLUX1_pro'
    fig_single_pro = display_single_model_images(df, image_folder, model_to_use=model_to_use)

    model_to_use = 'FLUX1_dev'
    fig_single_dev = display_single_model_images(df, image_folder, model_to_use=model_to_use)

    model_to_use = 'FLUX1_schnell'
    fig_single_schnell = display_single_model_images(df, image_folder, model_to_use=model_to_use)

    model_to_use = 'SDXL'
    fig_single_sdxl = display_single_model_images(df, image_folder, model_to_use=model_to_use)

    model_to_use = 'DALLE3'
    fig_single_dalle3 = display_single_model_images(df, image_folder, model_to_use=model_to_use)

    # Display multiple random images for each model
    models_names = ['FLUX1_pro', 'FLUX1_dev', 'FLUX1_schnell', 'SDXL', 'DALLE3']
    num_cols = 8
    fig_all = display_multi_model_images(df, image_folder, num_cols, models_names=models_names)

    models_names = ['FLUX1_pro', 'FLUX1_dev', 'FLUX1_schnell', 'SDXL']
    num_cols = 6
    fig_good_ones = display_multi_model_images(df, image_folder, num_cols, models_names=models_names)

    models_names = ['FLUX1_pro', 'FLUX1_dev', 'FLUX1_schnell']
    num_cols = 4
    fig_flux = display_multi_model_images(df, image_folder, num_cols, models_names=models_names)

    models_names = ['FLUX1_schnell', 'SDXL']
    num_cols = 4
    fig_bulk = display_multi_model_images(df, image_folder, num_cols, models_names=models_names)

    #%% Optionally, save the figures

    # save_figures = False
    save_figures = True

    if save_figures:
        output_folder_path = "figures"
        os.makedirs(output_folder_path, exist_ok=True)

        fig_distribution.savefig(os.path.join(output_folder_path, 'model_distribution.jpg'), bbox_inches='tight')
        fig_prompt_lengths.savefig(os.path.join(output_folder_path, 'prompt_lengths_distribution.jpg'), bbox_inches='tight')

        fig_single_pro.savefig(os.path.join(output_folder_path, 'FLUX1_pro_images_with_prompts.jpg'), bbox_inches='tight')
        fig_single_dev.savefig(os.path.join(output_folder_path, 'FLUX1_dev_images_with_prompts.jpg'), bbox_inches='tight')
        fig_single_schnell.savefig(os.path.join(output_folder_path, 'FLUX1_schnell_images_with_prompts.jpg'), bbox_inches='tight')
        fig_single_sdxl.savefig(os.path.join(output_folder_path, 'SDXL_images_with_prompts.jpg'), bbox_inches='tight')
        fig_single_dalle3.savefig(os.path.join(output_folder_path, 'DALLE3_images_with_prompts.jpg'), bbox_inches='tight')

        fig_all.savefig(os.path.join(output_folder_path, 'all_model_images.jpg'), bbox_inches='tight')
        fig_good_ones.savefig(os.path.join(output_folder_path, 'good_model_images.jpg'), bbox_inches='tight')
        fig_flux.savefig(os.path.join(output_folder_path, 'flux_images.jpg'), bbox_inches='tight')
        fig_bulk.savefig(os.path.join(output_folder_path, 'FLUX1_schnell_SDXL_images.jpg'), bbox_inches='tight')

    #%% select open clip model to use for textual search

    model_type = 'OpenCLIP'
    model_type = 'SigLIP'

    if model_type == 'OpenCLIP':
        model_name = "OpenCLIP_ViT-H-14-378-quickgelu"
        base_model_name = "ViT-H-14-378-quickgelu"
    elif model_type == 'SigLIP':
        model_name = "OpenCLIP_ViT-SO400M-14-SigLIP-384"
        base_model_name = "ViT-SO400M-14-SigLIP"

    # Extract features
    extract_features = False
    # extract_features = True
    if extract_features:
        extract_pretrained_features(dataset_path, model_to_use=model_name)

    # Collect previously extracted features
    image_features, image_filename_map = collect_pretrained_features_from_folder(dataset_path, model_name, normalize_features=True)
    print(f"Image features shape: {image_features.shape}")

    #%% Load the OpenCLIP model for text encoding

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = load_openclip_model(model_name, device)

    # define some text prompts for exploration
    text_prompts = {
        "age_male": ["baby boy", "boy todller", "child boy", "teenage boy", "adult male", "middle-aged adult male", "elderly male"],
        "age_female": ["baby girl", "girl todller", "child girl", "teenage girl", "adult female", "middle-aged adult female", "elderly female"],
        "expression": ["happy", "sad", "angry", "surprised", "neutral", "disgusted", "fearful", "tounge out"],
        "sex": ["male", "female", "non-binary person"],
        "ethnicity": ["Caucasian", "African", "Asian", "Hispanic", "Middle Eastern", "Scandinavian", "Native American"],
        "hair_color": ["black hair", "brown hair", "blonde hair", "red hair", "gray hair", "bald", "blue hair", "green hair", "pink hair"],
        "hats": ["baseball cap", "fedora", "beanie", "top hat", "cowboy hat", "sun hat"],
        "glasses": ["reading glasses", "sunglasses", "round glasses", "square glasses"],
        "accessories": ["earrings", "necklace", "bandana", "hat", "tie", "scarf", "headphones", "sunglasses"],
        "eye_color": ["blue eyes", "green eyes", "brown eyes", "yellow eyes", "hazel eyes", "red eyes"],
    }

    # define prefixes for each category
    prefixes = {
        "age_male": "A photo of a {prompt}",
        "age_female": "A photo of a {prompt}",
        "expression": "A photo of a person who is {prompt}",
        "sex": "A photo of a {prompt}",
        "ethnicity": "A photo of a {prompt} person",
        "hair_color": "A photo of a person with {prompt}",
        "hats": "A photo of a person wearing a {prompt}",
        "glasses": "A photo of a person wearing {prompt}",
        "accessories": "A photo of a person with {prompt}",
        "eye_color": "A photo of a person with {prompt}",
    }

    # encode the text prompts using the OpenCLIP text encoder
    tokenizer = open_clip.get_tokenizer(base_model_name)
    encoded_text_prompts = {}
    for category, prompts in text_prompts.items():
        prefix = prefixes.get(category, "A photo of a person with {prompt}")
        formatted_prompts = [prefix.format(prompt=prompt) for prompt in prompts]
        tokens = tokenizer(formatted_prompts).to(device)
        with torch.no_grad():
            encoded_text_prompts[category] = model.encode_text(tokens).float().cpu().numpy()
        encoded_text_prompts[category] /= np.linalg.norm(encoded_text_prompts[category], axis=1)[:, np.newaxis]

    # Calculate similarities between the texts and images and visualize results
    def visualize_category(category, image_features, encoded_prompts, prompts, num_cols=5, plot_boxplot=False):
        similarities = np.dot(image_features, encoded_prompts.T)
        
        if plot_boxplot:
            plt.figure(figsize=(12, 6))
            plt.boxplot(similarities)
            plt.title(f"Distribution of {category}")
            plt.xticks(range(1, len(prompts) + 1), prompts, rotation=45)
            plt.ylabel("Similarity Score")
            plt.tight_layout()
            plt.show()

        top_matches = np.argsort(similarities, axis=0)[::-1][:num_cols]
        
        fig = plt.figure(figsize=(3 * num_cols, 3 * len(prompts)))
        fig.patch.set_facecolor('black')
        
        for i, prompt in enumerate(prompts):
            for j in range(num_cols):
                ax = plt.subplot(len(prompts), num_cols, i * num_cols + j + 1)
                ax.set_facecolor('black')
                
                curr_image_filename = image_filename_map[top_matches[j, i]]
                base_filename = os.path.basename(curr_image_filename)
                img = Image.open(curr_image_filename)
                plt.imshow(img)
                plt.title(f"Match {j+1} for: {prompt}\n{base_filename}", fontsize=8, color='white')
                plt.axis('off')
        
        plt.tight_layout()

        return fig

    num_cols = 8
    for category, encoded_prompts in encoded_text_prompts.items():
        fig = visualize_category(category, image_features, encoded_prompts, text_prompts[category], num_cols=num_cols)

        if save_figures:
            figure_name_str = f'textual_search_1_{category}_top_{num_cols}_matches.jpg'
            fig.savefig(os.path.join(output_folder_path, figure_name_str), bbox_inches='tight')
        
    #%% make some textual searches on the dataset

    conditions_dict = {
        "Hair Color": {
            "text_prefix": "",
            "text_strings": ['white or gray hair', 'yellow or blond hair', 'green hair', 'blue hair', 'purple or pink hair', 'red or orange hair']
        },
        "Hair Style": {
            "text_prefix": "",
            "text_strings": ['straight hair', 'curly hair', 'high top hairstyle', 'bob-cut hairstyle', 'afro hairstyle']
        },
        "Hair Style x Sex": {
            "text_prefix": "woman with ",
            "text_strings": ['short blond hair', 'long blond hair', 'short red hair', 'long red hair', 'short black hair', 'long black hair']
        },
        "Makeup": {
            "text_prefix": "woman ",
            "text_strings": ['heavy makeup', 'without makeup', 'red lipstick', 'strong eyeliner', 'traditional makeup']
        },
        "Background Color": {
            "text_prefix": "",
            "text_strings": ['yellow background', 'green background', 'blue background', 'purple background', 'red background']
        },
        "Facial Features": {
            "text_prefix": "",
            "text_strings": ['reading glasses', 'sunglasses', 'bald', 'goatee', 'lipstick']
        },
        "Physical Characteristics": {
            "text_prefix": "",
            "text_strings": ['large or chiseled jaw', 'long white beard', 'fashionable beard', 'wide eyes', 'overweight or chubby']
        },
        "Expression": {
            "text_prefix": "",
            "text_strings": ['angry or enraged', 'surprised', 'smiling', 'sad or depressed', 'grim face', 'tounge out']
        },
        "Expression x Sex": {
            "text_prefix": "man ",
            "text_strings": ['angry or enraged', 'surprised', 'smiling', 'sad or depressed', 'grim face', 'tounge out']
        },
        "Ethnicity": {
            "text_prefix": "",
            "text_strings": ['asian', 'native american', 'african', 'persian', 'south-american', 'irish']
        },
        "Ethnicity x Age 1": {
            "text_prefix": "old age ",
            "text_strings": ['asian', 'native american', 'african', 'persian', 'south-american', 'irish']
        },
        "Ethnicity x Age 2": {
            "text_prefix": "typical adult ",
            "text_strings": ['asian', 'native american', 'african', 'persian', 'south-american', 'irish']
        },
        "Ethnicity x Age 3": {
            "text_prefix": "young child ",
            "text_strings": ['asian', 'native american', 'african', 'persian', 'south-american', 'irish']
        },
        "Age": {
            "text_prefix": "",
            "text_strings": ['10 month old baby', '2.5 year old toddler', 'small child', '16 year old teenager', '30 year old adult', 'wrinkly 70 year old senior']
        },
        "Age x Ethnicity x Sex 1": {
            "text_prefix": "asian female ",
            "text_strings": ['10 month old baby', '2.5 year old toddler', 'small child', '16 year old teenager', '30 year old adult', 'wrinkly 70 year old senior']
        },
        "Age x Ethnicity x Sex 2": {
            "text_prefix": "african male ",
            "text_strings": ['10 month old baby', '2.5 year old toddler', 'small child', '16 year old teenager', '30 year old adult', 'wrinkly 70 year old senior']
        },
        "Accessories": {
            "text_prefix": "person wearing ",
            "text_strings": ['earrings', 'necklace', 'bandana', 'hat', 'tie', 'scarf', 'headphones', 'sunglasses']
        },
        "Hats": {
            "text_prefix": "person wearing ",
            "text_strings": ['baseball cap', 'fedora', 'beanie', 'top hat', 'cowboy hat', 'sun hat']
        },
        "Jewelry": {
            "text_prefix": "person with ",
            "text_strings": ['gold chain', 'pearl necklace', 'earrings', 'diamond', 'crown']
        },
        "Face Pose": {
            "text_prefix": "person ",
            "text_strings": ['looking straight ahead', 'turned sideways', 'tilted upwards', 'tilted downwards', 'three-quarter view', 'profile view']
        },
        "Eye Gaze": {
            "text_prefix": "person ",
            "text_strings": ['looking directly at camera', 'looking to the left', 'looking up', 'looking down', 'eyes closed']
        },
        "Glasses Style": {
            "text_prefix": "person wearing ",
            "text_strings": ['round glasses', 'square glasses', 'cat-eye glasses', 'rimless glasses', 'aviator sunglasses', 'sport sunglasses']
        },
        "Facial Hair": {
            "text_prefix": "man with ",
            "text_strings": ['full beard', 'mustache', 'goatee', 'sideburns', 'stubble', 'shaved face']
        },
        "Lighting": {
            "text_prefix": "",
            "text_strings": ['side light with shadows', 'spotlight', 'soft lighting', 'back lighting', 'golden hour', 'blue hour lighting', 'studio lighting']
        },
        "Background": {
            "text_prefix": "",
            "text_strings": ['urban cityscape', 'natural landscape', 'stone wall background', 'wodden wall background', 'beach background', 'night background']
        },
        "Eye Color": {
            "text_prefix": "person with ",
            "text_strings": ['blue eyes', 'green eyes', 'brown eyes', 'yellow eyes', 'hazel eyes', 'red eyes']
        },
        "bad things": {
            "text_prefix": "",
            "text_strings": ['blurred', 'statue', 'two people', 'back of head', 'hand covering face', 'cat', 'dog', 'animal']
        },
    }

    for selected_condition in conditions_dict.keys():
        print(selected_condition)

        text_prefix = conditions_dict[selected_condition]['text_prefix']
        text_strings = conditions_dict[selected_condition]['text_strings']

        # will randomly display "num_top_images_to_show" among the top "num_top_image_candidates" best matching queries
        num_top_images_to_show = len(text_strings) + 4
        num_top_images_to_show = min(max(4, num_top_images_to_show), 10)
        num_top_image_candidates = int(1.0 * num_top_images_to_show)

        title_fontsize = 20

        # attach prefix and extract text features
        text_strings_full = [(text_prefix + x) for x in text_strings]
        tokenized_text_samples = tokenizer(text_strings_full).to(device)
        with torch.no_grad():
            openclip_text_features = model.encode_text(tokenized_text_samples).float().cpu().numpy()
        openclip_text_features /= np.linalg.norm(openclip_text_features, axis=1)[:, np.newaxis]  # normalize to unit norm

        # perform inner product to get image-text similarity score
        image_text_similarity = np.dot(image_features, openclip_text_features.T)

        num_rows = len(text_strings)
        num_cols = num_top_images_to_show

        fig, ax = plt.subplots(nrows=num_rows, ncols=num_cols, figsize=(5 * num_cols, 6 * num_rows))
        fig.patch.set_facecolor('0.0')
        fig.subplots_adjust(left=0.003, right=0.997, bottom=0.01, top=0.91, hspace=0.16, wspace=0.04)
        fig.suptitle(f'Image textual search using OpenCLIP features from synthetic dataset\nCondition: {selected_condition}\nPrefix: "{text_prefix}"', fontsize=30, color='white')

        all_basenames = []
        for row_ind, q_str in enumerate(text_strings):
            # get top "num_top_image_candidates" matching queries sorted from best matching downward
            query_best_inds = list(np.argsort(image_text_similarity[:,row_ind])[-num_top_image_candidates:])
            query_best_inds.reverse()
            # randomly select "num_top_images_to_show" from that list
            query_best_inds = np.random.choice(query_best_inds, size=num_top_images_to_show, replace=False)

            for col_ind in range(num_cols):
                curr_row_filename = image_filename_map[query_best_inds[col_ind]]
                curr_basename = os.path.basename(curr_row_filename)
                curr_image = Image.open(curr_row_filename).convert("RGB")
                ax[row_ind,col_ind].imshow(curr_image)
                ax[row_ind,col_ind].set_axis_off()
                ax[row_ind,col_ind].set_title(f"'{q_str}'\n{curr_basename}", fontsize=title_fontsize, color='white')

                all_basenames.append(curr_basename)

        if save_figures:
            figure_name_str = f'textual_search_2_{selected_condition}_top_{num_top_images_to_show}_matches.jpg'
            fig.savefig(os.path.join(output_folder_path, figure_name_str), bbox_inches='tight')


#%%




