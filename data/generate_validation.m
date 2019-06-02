clear;close all;
%% settings
size_input = 64;
size_label = 64;
image_dim = 3;

%%% generate 1 h5 file that contains 500 image patches for validation
num_h5_files = 1;
num_patches = 20;
%%%


%% initialization
data = zeros(size_input, size_input, image_dim, 1);
label = zeros(size_label, size_label, image_dim, 1);


%% generate data
listing = dir('./rainy_image_dataset/input_val/*.png');
for j = 1:num_h5_files
    count = 0;
    savepath = './h5data/validation.h5';
    
    for i = 1 : num_patches
        
        r_idx = random('unid', size(listing, 1));

        im_input = imread(strcat('./rainy_image_dataset/input_val/', listing(r_idx).name));  % input: rainy image
        im_input = double(im_input)./255.0;

        im_label = imread(strcat('./rainy_image_dataset/label_val/norain', listing(r_idx).name(6:end)));  % label: clean image
        im_label = double(im_label)./255.0;
        
        
        orig_img_size = size(im_label);
        x = random('unid', orig_img_size(1) - size_input + 1);
        y = random('unid', orig_img_size(2) - size_input + 1);
        
        subim_input = im_input(x : x+size_input-1, y : y+size_input-1,:);
        subim_label = im_label(x : x+size_input-1, y : y+size_input-1,:);
        
        count = count + 1;
        data(:, :, 1:image_dim, count) = flip(imrotate(subim_input,270),2);
        label(:, :, 1:image_dim, count) = flip(imrotate(subim_label,270),2);
        
        
    end
    
    order = randperm(count);
    data = data(:, :, 1:image_dim, order);
    label = label(:, :, 1:image_dim, order);
    
    %% writing to HDF5
    chunksz = 20;
    created_flag = false;
    totalct = 0;
    
    for batchno = 1:floor(count/chunksz)
        last_read = (batchno-1)*chunksz;
        batchdata = data(:,:,1:image_dim,last_read+1:last_read+chunksz);
        batchlabs = label(:,:,1:image_dim,last_read+1:last_read+chunksz);
        
        startloc = struct('dat',[1,1,1,totalct+1], 'lab', [1,1,1,totalct+1]);
        curr_dat_sz = store2hdf5(savepath, batchdata, batchlabs, ~created_flag, startloc, chunksz);
        created_flag = true;
        totalct = curr_dat_sz(end);
    end
    h5disp(savepath);
end
