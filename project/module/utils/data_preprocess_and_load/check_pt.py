import os
import torch
from tqdm import tqdm
from multiprocessing import Pool, Manager

def process_folder(args):
    folder, queue = args
    for file in os.listdir(folder):
        if file.endswith('.pt'):
            file_path = os.path.join(folder, file)
            try:
                data = torch.load(file_path)
            except Exception as e:
                print(f"Failed to read {file_path}: {e}")
    queue.put(1)

def read_pt_files(base_dir, num_processes):
    folders = [os.path.join(base_dir, folder) for folder in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, folder))]
    
    manager = Manager()
    queue = manager.Queue()
    pool = Pool(processes=num_processes)
    
    args = [(folder, queue) for folder in folders]
    
    pool.map_async(process_folder, args)
    
    with tqdm(total=len(folders), desc="Processing folders") as pbar:
        for _ in range(len(folders)):
            queue.get()
            pbar.update(1)
    
    pool.close()
    pool.join()

base_directory = 'data/UKB_MNI_to_TRs_minmax/img'
num_processes = 16
read_pt_files(base_directory, num_processes)
