from monai.transforms import LoadImage
import torch
import os
import time
from multiprocessing import Process, Queue

def select_middle_96(vector):
    start_index, end_index = [], []
    for i in range(3):
        if vector.shape[i] > 96:
            start_index.append((vector.shape[i] - 96) // 2)
            end_index.append(start_index[-1] + 96)
        else:
            start_index.append(0)
            end_index.append(-1)

    if len(vector.shape) == 3:
        result = vector[start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2]]
    elif len(vector.shape) == 4:
        result = vector[start_index[0]:end_index[0], start_index[1]:end_index[1], start_index[2]:end_index[2], :]
    
    return result

def read_data(filename,load_root,save_root,subj_name,count,queue=None,scaling_method=None, fill_zeroback=False):
    print("processing: " + filename, flush=True)
    path = os.path.join(load_root, filename)
    try:
        # load each nifti file
        data = LoadImage()(path)
    except:
        print('open failed')
        return None
    
    #change this line according to your file names
    save_dir = os.path.join(save_root, subj_name)
    isExist = os.path.exists(save_dir)
    if not isExist:
        os.makedirs(save_dir)
    
    data = select_middle_96(data)
    background = data==0
    
    if scaling_method == 'z-norm':
        global_mean = data[~background].mean()
        global_std = data[~background].std()
        data_temp = (data - global_mean) / global_std
    elif scaling_method == 'minmax':
        data_temp = (data - data[~background].min()) / (data[~background].max() - data[~background].min())

    data_global = torch.empty(data.shape)
    data_global[background] = data_temp[~background].min() if not fill_zeroback else 0 
    data_global[~background] = data_temp[~background]

    # save volumes one-by-one in fp16 format.
    data_global = data_global.type(torch.float16)
    data_global_split = torch.split(data_global, 1, 3)
    for i, TR in enumerate(data_global_split):
        torch.save(TR.clone(), os.path.join(save_dir,"frame_"+str(i)+".pt"))
    
    os.remove(path)


def main():
    parser = argparse.ArgumentParser(description='Process image data.')
    parser.add_argument('--load_root', type=str, required=True, help='Root directory to load data from')
    parser.add_argument('--save_root', type=str, required=True, help='Root directory to save data to')
    args = parser.parse_args()

    load_root = args.load_root
    save_root = args.save_root
    scaling_method = 'z-norm'
    
    filenames = os.listdir(load_root)
    os.makedirs(os.path.join(save_root, 'img'), exist_ok = True)
    os.makedirs(os.path.join(save_root, 'metadata'), exist_ok = True) # locate your metadata file at this folder 
    save_root = os.path.join(save_root, 'img')
    
    finished_samples = os.listdir(save_root)
    queue = Queue() 
    count = 0
    for filename in sorted(filenames):
        subj_name = filename[:-7] 

        expected_seq_length = 1200
        if (subj_name not in finished_samples) or (len(os.listdir(os.path.join(save_root, subj_name))) < expected_seq_length):
            try:
                count+=1
                p = Process(target=read_data, args=(filename,load_root,save_root,subj_name,count,queue,scaling_method))
                p.start()
                if count % 16 == 0: # requires more than 16 cpu cores for parallel processing
                    p.join()
            except Exception:
                print('encountered problem with'+filename)
                print(Exception)
        else:
            path = os.path.join(load_root, filename)
            save_dir = os.path.join(save_root, subj_name)
            print(f'{subj_name} has {len(os.listdir(os.path.join(save_root, subj_name)))} slices, save_dir is {save_dir}')
            os.remove(path)


if __name__=='__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')    
