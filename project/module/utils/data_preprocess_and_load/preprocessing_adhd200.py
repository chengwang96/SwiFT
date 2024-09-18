from monai.transforms import LoadImage
import torch
import os
import time
from multiprocessing import Process, Queue
import torch.nn.functional as F


def select_middle_96(vector):
    max_dim = max(vector.shape[:-1])
    resize_radio = 96 / max_dim
    new_size = (int(vector.shape[0] * resize_radio), int(vector.shape[1] * resize_radio), int(vector.shape[2] * resize_radio))

    if len(vector.shape) == 4:
        vector_permuted = vector.permute(3, 0, 1, 2)
        vector_unsqueezed = vector_permuted.unsqueeze(0)
    elif len(vector.shape) == 3:
        vector_unsqueezed = vector.unsqueeze(0).unsqueeze(0)
    output_tensor = F.interpolate(vector_unsqueezed, size=new_size, mode='trilinear', align_corners=True)
    if len(vector.shape) == 4:
        vector_squeezed = output_tensor.squeeze()
        vector = vector_squeezed.permute(1, 2, 3, 0)
    elif len(vector.shape) == 3:
        vector = output_tensor.squeeze()

    return vector


def read_data(filename, load_root, save_root, subj_name, count, queue=None, scaling_method=None, fill_zeroback=False):
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
    
    # change this line according to your dataset
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
    # data_temp[~background].min() is expected to be 0 for scaling_method == 'minmax', and minimum z-value for scaling_method == 'z-norm'
    data_global[~background] = data_temp[~background]

    # save volumes one-by-one in fp16 format.
    data_global = data_global.type(torch.float16)
    data_global_split = torch.split(data_global, 1, 3)
    for i, TR in enumerate(data_global_split):
        torch.save(TR.clone(), os.path.join(save_dir, "frame_"+str(i)+".pt"))


def main():
    load_root = './data/adhd200' # This folder should have fMRI files in nifti format with subject names. Ex) sub-01.nii.gz 
    save_root = f'./data/ADHD200_MNI_to_TRs_minmax'
    scaling_method = 'z-norm' # choose either 'z-norm'(default) or 'minmax'.

    # make result folders
    filenames = os.listdir(load_root)
    os.makedirs(os.path.join(save_root, 'img'), exist_ok = True)
    os.makedirs(os.path.join(save_root, 'metadata'), exist_ok = True) # locate your metadata file at this folder 
    save_root = os.path.join(save_root, 'img')
    
    finished_samples = os.listdir(save_root)
    queue = Queue() 
    count = 0
    for filename in sorted(filenames):
        if not filename.endswith('run1.nii.gz'):
            continue
    
        subj_name = filename.split('_')[2]
        expected_seq_length = 120

        # fill_zeroback = False
        # print("processing: " + filename, flush=True)
        # path = os.path.join(load_root, filename)
        # try:
        #     data = LoadImage()(path)
        # except:
        #     print('open failed')
        #     return None
        
        # data = select_middle_96(data)
        
        # background = data==0
        
        # if scaling_method == 'z-norm':
        #     global_mean = data[~background].mean()
        #     global_std = data[~background].std()
        #     data_temp = (data - global_mean) / global_std
        # elif scaling_method == 'minmax':
        #     data_temp = (data - data[~background].min()) / (data[~background].max() - data[~background].min())

        # data_global = torch.empty(data.shape)
        # data_global[background] = data_temp[~background].min() if not fill_zeroback else 0
        # data_global[~background] = data_temp[~background]

        # data_global = data_global.type(torch.float16)
        # data_global_split = torch.split(data_global, 1, 3)
        # import ipdb; ipdb.set_trace()

        if (subj_name not in finished_samples) or (len(os.listdir(os.path.join(save_root, subj_name))) < expected_seq_length):
            try:
                count+=1
                # read_data(filename, load_root, save_root, subj_name, count, queue, scaling_method)
                p = Process(target=read_data, args=(filename, load_root, save_root, subj_name, count, queue, scaling_method))
                p.start()
                if count % 64 == 0: # requires more than 16 cpu cores for parallel processing
                    p.join()
            except Exception:
                print('encountered problem with'+filename)
                print(Exception)
        else:
            path = os.path.join(load_root, filename)
            save_dir = os.path.join(save_root, subj_name)
            print('{} has {} slices, save_dir is {}'.format(subj_name, len(os.listdir(os.path.join(save_root, subj_name))), save_dir))
            # import ipdb; ipdb.set_trace()
            # os.remove(path)


if __name__=='__main__':
    start_time = time.time()
    main()
    end_time = time.time()
    print('\nTotal', round((end_time - start_time) / 60), 'minutes elapsed.')    
