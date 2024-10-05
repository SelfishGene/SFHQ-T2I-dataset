#%% Imports

import os
import time
import shutil
import pandas as pd
from tqdm import tqdm

#%% Helper functions

def merge_datasets(source_folders, output_folder):
    # Create output folders
    output_image_folder = os.path.join(output_folder, "images")
    os.makedirs(output_image_folder, exist_ok=True)
    
    # Initialize a list to store all metadata
    all_metadata = []
    
    # Initialize counters for each model
    model_counters = {
        "SDXL": 1,
        "DALLE3": 1,
        "FLUX1_pro": 1,
        "FLUX1_dev": 1,
        "FLUX1_schnell": 1
    }
    
    start_time = time.time()
    # Process each source folder
    for source_folder in source_folders:
        source_image_folder = os.path.join(source_folder, "images")
        source_csv_path = os.path.join(source_folder, "SFHQ_T2I_dataset.csv")
        
        # Read the CSV file
        df = pd.read_csv(source_csv_path)
        
        # Get all image files in the source folder
        image_files = [f for f in os.listdir(source_image_folder) if f.endswith('.jpg')]
        
        print(f"Processing '{source_folder}' ...")
        for image_file in tqdm(image_files):
            # Find the corresponding metadata
            metadata_row = df[df['image_filename'] == image_file]
            
            if not metadata_row.empty:
                model = metadata_row['model_used'].iloc[0]
                new_image_name = f"{model}_image_{model_counters[model]:07d}.jpg"
                
                # Copy and rename the image
                shutil.copy(
                    os.path.join(source_image_folder, image_file),
                    os.path.join(output_image_folder, new_image_name)
                )
                
                # Update metadata
                new_metadata = {
                    'image_filename': new_image_name,
                    'model_used': model,
                    'text_prompt': metadata_row['text_prompt'].iloc[0],
                    'configs': metadata_row['configs'].iloc[0]
                }
                all_metadata.append(new_metadata)
                
                # Increment the counter for this model
                model_counters[model] += 1
    
    # Create the final dataframe and save it
    final_df = pd.DataFrame(all_metadata)
    final_df = final_df.sort_values(by='image_filename').reset_index(drop=True)
    final_csv_path = os.path.join(output_folder, "SFHQ_T2I_dataset.csv")
    final_df.to_csv(final_csv_path, index=False)
    
    total_duration_minutes = (time.time() - start_time) / 60
    print(f"\nDataset merging completed! Total duration: {total_duration_minutes:.2f} minutes")
    print(f"Total images in the merged dataset per model:")
    for model, count in model_counters.items():
        print(f"- {count - 1} {model} images")
    print(f"Combined total of images: {len(final_df)}")
    print(f"\nMetadata saved to '{final_csv_path}'")


#%% Usage

if __name__ == "__main__":

    source_folders = [
        r"datasample_001",
        r"datasample_002",
        r"datasample_003",
    ]

    output_folder = r"merged_clean_dataset"
    
    merge_datasets(source_folders, output_folder)

#%%

