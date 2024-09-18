from monai.transforms import LoadImage
import torch
import os
import time
from multiprocessing import Process, Queue
import torch.nn.functional as F
import pandas as pd


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


def read_data(metadata, filename, load_root, save_root, subj_name, count, queue=None, scaling_method=None, fill_zeroback=False):
    expected_seq_length = 20
    print("processing: " + filename, flush=True)
    path = os.path.join(load_root, filename)
    
    try:
        data = LoadImage()(path)
    except:
        print('open failed')
        return None

    data = select_middle_96(data)
    background = data == 0

    if scaling_method == 'z-norm':
        global_mean = data[~background].mean()
        global_std = data[~background].std()
        data_temp = (data - global_mean) / global_std
    elif scaling_method == 'minmax':
        data_temp = (data - data[~background].min()) / (data[~background].max() - data[~background].min())

    data_global = torch.empty(data.shape)
    data_global[background] = data_temp[~background].min() if not fill_zeroback else 0
    data_global[~background] = data_temp[~background]

    relevant_rows = metadata[metadata['subject_id'].str.startswith(subj_name[:13])]
    data_global = data_global[:, :, :, 7:]
    
    for _, row in relevant_rows.iterrows():
        onset = int((row['onset'] - 33) / 3)
        duration = int((row['duration']) / 3)
        subject_id = row['subject_id']
        
        data_slice = data_global[:, :, :, onset:(onset + duration)]
        h, w, l, t = data_slice.shape
        data_slice = data_slice.reshape(-1, t)
        data_slice_interpolated = F.interpolate(data_slice.unsqueeze(0).unsqueeze(0), size=(h*w*l, expected_seq_length), mode='bilinear', align_corners=False)
        data_slice_interpolated = data_slice_interpolated.reshape(h, w, l, expected_seq_length)
        data_slice_interpolated = data_slice_interpolated.type(torch.float16)
        
        data_global_split = torch.split(data_slice_interpolated, 1, 3)
        save_dir = os.path.join(save_root, subject_id)
        os.makedirs(save_dir, exist_ok=True)

        for i, TR in enumerate(data_global_split):
            torch.save(TR.clone(), os.path.join(save_dir, "frame_"+str(i)+".pt"))


def main():
    load_root = './data/GOD_clean'
    save_root = f'./data/GOD_MNI_to_TRs_minmax'
    scaling_method = 'z-norm'

    csv_path = './data/GOD_clean/god_label.csv'
    metadata = pd.read_csv(csv_path) 

    filenames = os.listdir(load_root)
    os.makedirs(os.path.join(save_root, 'img'), exist_ok = True)
    os.makedirs(os.path.join(save_root, 'metadata'), exist_ok = True)
    save_root = os.path.join(save_root, 'img')
    
    finished_samples = os.listdir(save_root)
    queue = Queue() 
    count = 0
    for filename in sorted(filenames):
        if not filename.endswith('preproc.nii.gz') or not 'perception' in filename:
            continue
    
        subj_name = filename[:6] + '_' + filename.split('perception_')[1][:6]
        expected_seq_length = 20

        # fill_zeroback = False
        # print("processing: " + filename, flush=True)
        # path = os.path.join(load_root, filename)
        
        # try:
        #     data = LoadImage()(path)
        # except:
        #     print('open failed')
        #     continue

        # data = select_middle_96(data)
        # background = data == 0

        # if scaling_method == 'z-norm':
        #     global_mean = data[~background].mean()
        #     global_std = data[~background].std()
        #     data_temp = (data - global_mean) / global_std
        # elif scaling_method == 'minmax':
        #     data_temp = (data - data[~background].min()) / (data[~background].max() - data[~background].min())

        # data_global = torch.empty(data.shape)
        # data_global[background] = data_temp[~background].min() if not fill_zeroback else 0
        # data_global[~background] = data_temp[~background]

        # relevant_rows = metadata[metadata['subject_id'].str.startswith(subj_name[:13])]
        # data_global = data_global[:, :, :, 7:]
        
        # for _, row in relevant_rows.iterrows():
        #     onset = int((row['onset'] - 33) / 3)
        #     duration = int((row['duration']) / 3)
        #     subject_id = row['subject_id']
            
        #     data_slice = data_global[:, :, :, onset:(onset + duration)]
        #     h, w, l, t = data_slice.shape
        #     data_slice = data_slice.reshape(-1, t)
        #     data_slice_interpolated = F.interpolate(data_slice.unsqueeze(0).unsqueeze(0), size=(h*w*l, expected_seq_length), mode='bilinear', align_corners=False)
        #     data_slice_interpolated = data_slice_interpolated.reshape(h, w, l, expected_seq_length)
        #     data_slice_interpolated = data_slice_interpolated.type(torch.float16)
            
        #     data_global_split = torch.split(data_slice_interpolated, 1, 3)
            
        #     save_dir = os.path.join(save_root, subject_id)
        #     os.makedirs(save_dir, exist_ok=True)
        #     import ipdb; ipdb.set_trace()

        #     for i, TR in enumerate(data_global_split):
        #         torch.save(TR.clone(), os.path.join(save_dir, "frame_"+str(i)+".pt"))

        if (subj_name not in finished_samples) or (len(os.listdir(os.path.join(save_root, subj_name))) < expected_seq_length):
            try:
                count+=1
                # read_data(filename, load_root, save_root, subj_name, count, queue, scaling_method)
                p = Process(target=read_data, args=(metadata, filename, load_root, save_root, subj_name, count, queue, scaling_method))
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
