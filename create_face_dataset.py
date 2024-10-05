#%% Imports

import os
import re
import io
import time
import json
import base64
import random
import requests
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
import asyncio
from tqdm.asyncio import tqdm as async_tqdm
import fal_client
from openai import OpenAI
from stability_sdk import client as sd_client
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation

from face_prompt_utils import generate_face_prompt

#%% API Key Configurations

STABILITY_API_KEY = 'sk-abcdefghigklmnopqrstuvwxyz1234567890abcdefghigkl'
OPENAI_API_KEY = 'sk-abcd1234efgh5678ijkl9012mnop3456qrst7890uvwx1234yzab5678cdef9012ghij3456klmn7890opqr1234stuv'
FAL_API_KEY = 'abcdefgh-ijkl-mnop-qrst-uvwxyz123456:a1234567890abcdefghijklmnopqrabc'

def save_env_file():
    """Create a .env file with API keys if it doesn't exist."""

    env_file = '.env'
    if not os.path.exists(env_file):
        print("Creating .env file...")
        with open(env_file, 'w') as f:
            f.write(f"STABILITY_API_KEY={STABILITY_API_KEY}\n")
            f.write(f"OPENAI_API_KEY={OPENAI_API_KEY}\n")
            f.write(f"FAL_KEY={FAL_API_KEY}\n")
        print(f".env file created at {os.path.abspath(env_file)}")
        print("Please edit the .env file with your actual API keys before running the script again.")
        exit()

def setup_api_keys():
    # Load the .env file
    load_dotenv()
    
    # Check if all required keys are present
    required_keys = ['STABILITY_API_KEY', 'OPENAI_API_KEY', 'FAL_KEY']
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"Error: The following API keys are missing in the .env file: {', '.join(missing_keys)}")
        print("Please add them to the .env file and run the script again.")
    else:
        print("API keys loaded successfully.")

# Call the setup function at the beginning of the script
save_env_file()
setup_api_keys()

# print to the screen all the API keys that were loaded
key_name_list = ['STABILITY_API_KEY', 'OPENAI_API_KEY', 'FAL_KEY']
for key_name in key_name_list:
    print(f'{key_name} = {os.getenv(key_name)}')

#%% API clients

openai_client = OpenAI(api_key = os.getenv('OPENAI_API_KEY'))

#%% Constants

SDXL_STYLES = [
    "3d-model", "analog-film", "anime", "cinematic", "comic-book", "digital-art",
    "enhance", "fantasy-art", "isometric", "line-art", "low-poly", "modeling-compound",
    "neon-punk", "origami", "photographic", "pixel-art", "tile-texture"
]

SDXL_STYLES = ["analog-film", "cinematic", "photographic", "enhance"]
DALLE3_IMAGE_SIZES = ["1024x1024", "1024x1792", "1792x1024"]
DALLE3_STYLES = ["vivid", "natural"]
DALLE3_QUALITIES = ["standard", "hd"]
FLUX_IMAGE_SIZES = ["square_hd", "square", "portrait_4_3", "portrait_16_9", "landscape_4_3", "landscape_16_9"]

FLUX_API_MODEL_NAME_DICT = {
    'FLUX1_pro': 'fal-ai/flux-pro',
    'FLUX1_dev': 'fal-ai/flux/dev',
    'FLUX1_schnell': 'fal-ai/flux/schnell'
}

#%% Helper functions

def generate_image_SDXL(prompt, engine_id, cfg_scale, steps, seed, style_preset):
    stability_api = sd_client.StabilityInference(key=os.getenv('STABILITY_API_KEY'), engine=engine_id)

    params = {
        "prompt": prompt,
        "cfg_scale": cfg_scale,
        "steps": steps,
        "seed": seed,
        "style_preset": style_preset
    }

    response = stability_api.generate(**params)

    for resp in response:
        for artifact in resp.artifacts:
            if artifact.type == generation.ARTIFACT_IMAGE:
                return Image.open(io.BytesIO(artifact.binary))

    return None

def generate_image_DALLE3(prompt, size='1024x1024', quality='standard', style='vivid', response_format='url'):
    
    response = openai_client.images.generate(
        model="dall-e-3",
        prompt=prompt,
        size=size,
        quality=quality,
        style=style,
        response_format=response_format
    )
    
    if response_format == "b64_json":
        image_data = base64.b64decode(response.data[0].b64_json)
        image_PIL = Image.open(io.BytesIO(image_data))
    elif response_format == "url":
        image_url = response.data[0].url
        image_data = io.BytesIO(requests.get(image_url).content)
        image_PIL = Image.open(image_data)

    revised_prompt = response.data[0].revised_prompt

    return image_PIL, revised_prompt

def generate_image_FLUX(prompt, api_model_name, seed, num_inference_steps, image_size='square_hd', guidance_scale=3.5):
    
    if api_model_name == 'fal-ai/flux-pro':
        arguments = {
            "prompt": prompt,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "safety_tolerance": "5",
            "sync_mode": True
        }
    elif api_model_name == 'fal-ai/flux/dev':
        arguments = {
            "prompt": prompt,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "enable_safety_checker": False,
            "sync_mode": True
        }
    elif api_model_name == 'fal-ai/flux/schnell':
        arguments = {
            "prompt": prompt,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "enable_safety_checker": False,
            "sync_mode": True
        }

    handler = fal_client.submit(api_model_name, arguments=arguments)
    result = handler.get()

    image_url = result['images'][0]['url']
    if image_url.startswith('data:image/jpeg;base64,'):
        image_data = io.BytesIO(base64.b64decode(image_url.split(',')[1]))
    else:
        image_data = io.BytesIO(requests.get(image_url).content)

    image_PIL = Image.open(image_data)

    return image_PIL

async def generate_image_FLUX_async(prompt, api_model_name, seed, num_inference_steps=50, image_size='square_hd', guidance_scale=3.5):
    if api_model_name == 'fal-ai/flux-pro':
        arguments = {
            "prompt": prompt,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "safety_tolerance": "5",
            "sync_mode": False
        }
    elif api_model_name == 'fal-ai/flux/dev':
        arguments = {
            "prompt": prompt,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "guidance_scale": guidance_scale,
            "enable_safety_checker": False,
            "sync_mode": False
        }
    elif api_model_name == 'fal-ai/flux/schnell':
        arguments = {
            "prompt": prompt,
            "image_size": image_size,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
            "enable_safety_checker": False,
            "sync_mode": False
        }

    handler = await fal_client.submit_async(api_model_name, arguments=arguments)
    result = await handler.get()

    image_url = result['images'][0]['url']
    if image_url.startswith('data:image/jpeg;base64,'):
        image_data = io.BytesIO(base64.b64decode(image_url.split(',')[1]))
    else:
        image_data = io.BytesIO(requests.get(image_url).content)

    image_PIL = Image.open(image_data)

    return image_PIL

def generate_image_with_retry(generate_func, max_retries=2, **kwargs):
    for attempt in range(max_retries):
        try:
            return generate_func(**kwargs)
        except Exception as e:
            print(f"Error occurred: {e}")
            if attempt < max_retries - 1:
                wait_time = random.uniform(0.5, 2)
                print(f"Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                print("Max retries reached. Skipping this generation.")
                return None

def get_existing_image_count(image_folder, model_prefix):
    # Regular expression to match the number at the end of the filename
    pattern = re.compile(rf"{re.escape(model_prefix)}_image_(\d+)\.jpg")
    
    max_number = 0
    for filename in os.listdir(image_folder):
        match = pattern.match(filename)
        if match:
            number = int(match.group(1))
            max_number = max(max_number, number)
    
    return max_number

def create_dataset_SDXL(num_samples, image_folder, engine_id, steps, jpeg_quality=90):
    metadata = []
    total_time = 0
    start_index = get_existing_image_count(image_folder, "SDXL")
    
    with tqdm(total=num_samples, desc="Generating SDXL images") as pbar:
        for i in range(num_samples):
            prompt = get_random_prompt()
            style_preset = random.choice(SDXL_STYLES)
            seed = random.randint(0, 2**32 - 1)
            cfg_scale = random.randint(5, 8)
            
            start_time = time.time()
            image = generate_image_with_retry(
                generate_image_SDXL,
                prompt=prompt,
                engine_id=engine_id,
                cfg_scale=cfg_scale,
                steps=steps,
                seed=seed,
                style_preset=style_preset
            )
            end_time = time.time()
            
            if image:
                image_filename = f"SDXL_image_{start_index + i + 1:07d}.jpg"
                image_path = os.path.join(image_folder, image_filename)
                
                image.save(image_path, "JPEG", quality=jpeg_quality)
                
                configs = {
                    "engine_id": engine_id,
                    "cfg_scale": cfg_scale,
                    "steps": steps,
                    "seed": seed,
                    "style_preset": style_preset
                }
                
                metadata.append({
                    "image_filename": image_filename,
                    "model_used": "SDXL",
                    "text_prompt": prompt,
                    "configs": json.dumps(configs),
                })
                
                total_time += (end_time - start_time)
                pbar.update(1)
    
    print(f"SDXL: Generated {num_samples} images in {total_time/60:.2f} minutes (avg: {total_time/num_samples:.2f} seconds per image)")
    return pd.DataFrame(metadata)

def create_dataset_DALLE3(num_samples, image_folder, size, quality, jpeg_quality=90):
    metadata = []
    total_time = 0
    start_index = get_existing_image_count(image_folder, "DALLE3")
    
    with tqdm(total=num_samples, desc="Generating DALL-E 3 images") as pbar:
        for i in range(num_samples):
            prompt = get_random_prompt()
            style = random.choice(DALLE3_STYLES)
            
            start_time = time.time()
            result = generate_image_with_retry(
                generate_image_DALLE3,
                prompt=prompt,
                size=size,
                quality=quality,
                style=style
            )
            end_time = time.time()
            
            if result:
                image, revised_prompt = result
                image_filename = f"DALLE3_image_{start_index + i + 1:07d}.jpg"
                image_path = os.path.join(image_folder, image_filename)
                
                image.save(image_path, "JPEG", quality=jpeg_quality)
                
                configs = {
                    "size": size,
                    "quality": quality,
                    "style": style,
                    "orig_prompt": prompt
                }
                
                metadata.append({
                    "image_filename": image_filename,
                    "model_used": "DALLE3",
                    "text_prompt": revised_prompt,
                    "configs": json.dumps(configs),
                })
                
                total_time += (end_time - start_time)
                pbar.update(1)
    
    print(f"DALL-E 3: Generated {len(metadata)} images in {total_time/60:.2f} minutes (avg: {total_time/len(metadata):.2f} seconds per image)")
    return pd.DataFrame(metadata)

def create_dataset_FLUX(num_samples, flux_model, image_folder, num_inference_steps, image_size, jpeg_quality=90):
    metadata = []
    total_time = 0
    start_index = get_existing_image_count(image_folder, flux_model)
    
    flux_api_model_name = FLUX_API_MODEL_NAME_DICT[flux_model]

    with tqdm(total=num_samples, desc=f"Generating {flux_model} images") as pbar:
        for i in range(num_samples):
            prompt = get_random_prompt()
            seed = random.randint(0, 2**32 - 1)
            guidance_scale = random.uniform(2.5, 4.0) if random.random() < 0.5 else 3.5
            
            start_time = time.time()
            image = generate_image_with_retry(
                generate_image_FLUX,
                prompt=prompt,
                api_model_name=flux_api_model_name,
                seed=seed,
                num_inference_steps=num_inference_steps,
                image_size=image_size,
                guidance_scale=guidance_scale
            )
            end_time = time.time()
            
            if image:
                image_filename = f"{flux_model}_image_{start_index + i + 1:07d}.jpg"
                image_path = os.path.join(image_folder, image_filename)
                
                image.save(image_path, "JPEG", quality=jpeg_quality)
                
                if flux_model == 'FLUX1_pro':
                    configs = {
                        "image_size": image_size,
                        "num_inference_steps": num_inference_steps,
                        "seed": seed,
                        "guidance_scale": guidance_scale,
                        "safety_tolerance": "5",
                    }
                elif flux_model == 'FLUX1_dev':
                    configs = {
                        "image_size": image_size,
                        "num_inference_steps": num_inference_steps,
                        "seed": seed,
                        "guidance_scale": guidance_scale,
                        "enable_safety_checker": False,
                    }
                elif flux_model == 'FLUX1_schnell':
                    configs = {
                        "image_size": image_size,
                        "num_inference_steps": num_inference_steps,
                        "seed": seed,
                        "enable_safety_checker": False,
                    }

                metadata.append({
                    "image_filename": image_filename,
                    "model_used": flux_model,
                    "text_prompt": prompt,
                    "configs": json.dumps(configs),
                })
                
                total_time += (end_time - start_time)
                pbar.update(1)
    
    print(f"{flux_model}: Generated {len(metadata)} images in {total_time/60:.2f} minutes (avg: {total_time/len(metadata):.2f} seconds per image)")
    return pd.DataFrame(metadata)

async def create_dataset_FLUX_parallel(num_samples, flux_model, image_folder, num_inference_steps, image_size, jpeg_quality=90, max_concurrent_calls=5):
    dataset_start_time = time.time()

    metadata = []
    start_index = get_existing_image_count(image_folder, flux_model)    
    flux_api_model_name = FLUX_API_MODEL_NAME_DICT[flux_model]
    semaphore = asyncio.Semaphore(max_concurrent_calls)

    async def process_single_image(i):
        async with semaphore:
            prompt = get_random_prompt()
            seed = random.randint(0, 2**32 - 1)
            guidance_scale = random.uniform(2.5, 4.0) if random.random() < 0.5 else 3.5
            
            sample_start_time = time.time()
            try:
                image = await generate_image_FLUX_async(
                    prompt=prompt,
                    api_model_name=flux_api_model_name,
                    seed=seed,
                    num_inference_steps=num_inference_steps,
                    image_size=image_size,
                    guidance_scale=guidance_scale
                )
            except Exception as e:
                print(f"Error generating image for {flux_model}: {e}")
                return None
            sample_end_time = time.time()
            
            if image:
                image_filename = f"{flux_model}_image_{start_index + i + 1:07d}.jpg"
                image_path = os.path.join(image_folder, image_filename)
                
                image.save(image_path, "JPEG", quality=jpeg_quality)
                
                configs = {
                    "image_size": image_size,
                    "num_inference_steps": num_inference_steps,
                    "seed": seed,
                    "guidance_scale": guidance_scale,
                }
                if flux_model == 'FLUX1_pro':
                    configs["safety_tolerance"] = "5"
                elif flux_model in ['FLUX1_dev', 'FLUX1_schnell']:
                    configs["enable_safety_checker"] = False

                sample_durations_sec = sample_end_time - sample_start_time
                return {
                    "image_filename": image_filename,
                    "model_used": flux_model,
                    "text_prompt": prompt,
                    "configs": json.dumps(configs),
                }
            return None

    tasks = [process_single_image(i) for i in range(num_samples)]    
    results = await async_tqdm.gather(*tasks, desc=f"Generating {flux_model} images")
    metadata = [result for result in results if result is not None]
    total_time = time.time() - dataset_start_time
    print(f"{flux_model}: Generated {len(metadata)} images in {total_time/60:.2f} minutes (avg: {total_time/len(metadata):.2f} seconds per image)")
    
    return pd.DataFrame(metadata)

def update_csv(new_df, csv_path):
    if os.path.exists(csv_path):
        existing_df = pd.read_csv(csv_path)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined_df = new_df
    
    combined_df.to_csv(csv_path, index=False)
    return combined_df

def get_random_prompt():
    output_prompt = generate_face_prompt()
    return output_prompt

#%%

if __name__ == "__main__":

    # Explicit configuration variables
    # output_db_folder = r"datasample_001"
    output_db_folder = r"datasample_002"
    
    os.makedirs(output_db_folder, exist_ok=True)

    call_dev_pro_async = True

    # FLUX1.dev (about 1150 images per 1 hour when async is on, costs ~$29 per 1150 images)
    flux1_dev_samples = 10
    flux1_dev_config = {
        "image_size": "square_hd",
        "num_inference_steps": 50,
        'jpeg_quality': 90
    }

    # FLUX1.pro (about 1100 images per 1 hour when async is on, costs ~$55 per 1100 images)
    flux1_pro_samples = 10
    flux1_pro_config = {
        "image_size": "square_hd",
        "num_inference_steps": 50,
        'jpeg_quality': 90
    }

    # SDXL (about 550 images per 1 hour, costs ~$2 per 550 images)
    sdxl_samples = 10
    sdxl_config = {
        "engine_id": "stable-diffusion-xl-1024-v1-0",
        "steps": 70,
        'jpeg_quality': 90
    }

    # FLUX1.schnell (about 2000 images per 1 hour, costs ~$6 per 2000 images)
    flux1_schnell_samples = 10
    flux1_schnell_config = {
        "image_size": "square_hd",
        "num_inference_steps": 12,
        'jpeg_quality': 90
    }

    # DALL-E 3 (about 233 images per 1 hour, costs ~$8.6 per 233 images)
    dalle3_samples = 10
    dalle3_config = {
        "size": "1024x1024",
        "quality": "standard",
        'jpeg_quality': 90
    }

    # Create the mixed dataset
    image_folder = os.path.join(output_db_folder, "images")
    os.makedirs(image_folder, exist_ok=True)
    csv_path = os.path.join(output_db_folder, "SFHQ_T2I_dataset.csv")
    
    print("\nStarting image generation...\n")
    
    if call_dev_pro_async:
        max_concurrent_calls = 10

        loop = asyncio.get_event_loop()

        if flux1_dev_samples > 0:
            flux1_dev_df = loop.run_until_complete(create_dataset_FLUX_parallel(
                flux1_dev_samples, 'FLUX1_dev', image_folder, max_concurrent_calls=max_concurrent_calls, **flux1_dev_config
            ))
            combined_df = update_csv(flux1_dev_df, csv_path)
            print(f"CSV updated with {len(flux1_dev_df)} FLUX1_dev images")

        if flux1_pro_samples > 0:
            flux1_pro_df = loop.run_until_complete(create_dataset_FLUX_parallel(
                flux1_pro_samples, 'FLUX1_pro', image_folder, max_concurrent_calls=max_concurrent_calls, **flux1_pro_config
            ))
            combined_df = update_csv(flux1_pro_df, csv_path)
            print(f"CSV updated with {len(flux1_pro_df)} FLUX1_pro images")

        loop.close()
    else:
        if flux1_dev_samples > 0:
            flux1_dev_df = create_dataset_FLUX(flux1_dev_samples, 'FLUX1_dev', image_folder, **flux1_dev_config)
            combined_df = update_csv(flux1_dev_df, csv_path)
            print(f"CSV updated with {len(flux1_dev_df)} FLUX1_dev images")

        if flux1_pro_samples > 0:
            flux1_pro_df = create_dataset_FLUX(flux1_pro_samples, 'FLUX1_pro', image_folder, **flux1_pro_config)
            combined_df = update_csv(flux1_pro_df, csv_path)
            print(f"CSV updated with {len(flux1_pro_df)} FLUX1_pro images")

    if flux1_schnell_samples > 0:
        flux1_schnell_df = create_dataset_FLUX(flux1_schnell_samples, 'FLUX1_schnell', image_folder, **flux1_schnell_config)
        combined_df = update_csv(flux1_schnell_df, csv_path)
        print(f"CSV updated with {len(flux1_schnell_df)} FLUX1_schnell images")

    if sdxl_samples > 0:
        sdxl_df = create_dataset_SDXL(sdxl_samples, image_folder, **sdxl_config)
        combined_df = update_csv(sdxl_df, csv_path)
        print(f"CSV updated with {len(sdxl_df)} SDXL images")
    
    if dalle3_samples > 0:
        dalle3_df = create_dataset_DALLE3(dalle3_samples, image_folder, **dalle3_config)
        combined_df = update_csv(dalle3_df, csv_path)
        print(f"CSV updated with {len(dalle3_df)} DALLE3 images")
    
    print("\nDataset creation completed!\n")
    print(f"Total images in the dataset per model:")
    for model in ["SDXL", "DALLE3", "FLUX1_pro", "FLUX1_dev", "FLUX1_schnell"]:
        count = len(combined_df[combined_df['model_used'] == model])
        print(f"- {count} {model} images")
    print(f"Combined total of images: {len(combined_df)}")
    print(f"\nMetadata saved to 'SFHQ_T2I_dataset.csv'")


#%%

