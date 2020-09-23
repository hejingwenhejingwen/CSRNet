function generate_3D_val()

input_path = '../../datasets/CBSD68'
save_LR_root = '../../datasets/CBSD68_3D_val';


file_type = '.png';

% kernel
kernel_list = [0, 1, 4];
kernel_label_list = kernel_list*10;

% noise
noise_list = [0, 15, 50];
noise_label_list = noise_list;

% JPEG {None, [100, 10]}
jpeg_list = [102, 80, 10]; % 102 -> no JPEG
jpeg_label_list = 92 - (jpeg_list - 10);


for k_ind = 1:length(kernel_list)
    kernelwidth = kernel_list(k_ind);
    kernel_label = kernel_label_list(k_ind);
    if kernel_label > 0
        kernel = single(fspecial('gaussian', 21, kernelwidth));
    end 

    for s_ind = 1:length(noise_list)
        noiseSigma = noise_list(s_ind);
        noise_label = noise_label_list(s_ind);

        for q_ind = 1:length(jpeg_list)
            JPEG_Quality = jpeg_list(q_ind);
            jpeg_label = jpeg_label_list(q_ind);

            save_LR_path = fullfile(save_LR_root, ['blur' num2str(kernel_label) '_noise'  num2str(noiseSigma) '_jpeg' num2str(JPEG_Quality)]);

            if exist('save_LR_path', 'var')
                if exist(save_LR_path, 'dir')
                    disp(['It will cover ', save_LR_path]);
                else
                    mkdir(save_LR_path);
                end
            end
            
            mkdir(save_LR_path);

            randn('seed', 0);

            idx = 0;
            filepaths = dir(fullfile(input_path,'*.*'));
            for i = 1 : length(filepaths)
                [paths,imname,ext] = fileparts(filepaths(i).name);
                if isempty(imname)
                    disp('Ignore . folder.');
                elseif strcmp(imname, '.')
                    disp('Ignore .. folder.');
                else
                    idx = idx + 1;
                    str_rlt = sprintf('%d\t%s.\n', idx, imname);
                    fprintf(str_rlt);

                    % read image
                    img = im2double(imread(fullfile(input_path, [imname, ext])));
                    if kernel_label > 0
                        blurry_img = imfilter(img, double(kernel), 'replicate');
                    else
                        blurry_img = img;
                    end

                    % add noise
                    if noise_label > 0
                        noise = noiseSigma/255.*randn(size(blurry_img));
                        im_noise = single(blurry_img + noise);
                        im_noise = im2uint8(im_noise);
                    else
                        im_noise = blurry_img;
                    end
                   

                    if exist('save_LR_path', 'var')
                        if JPEG_Quality <= 100 
                            imwrite(im_noise, fullfile(save_LR_path, [imname '_' num2str(kernel_label, '%02d') num2str(noise_label, '%02d') num2str(jpeg_label, '%02d') '.jpg']), 'jpg', 'Quality', JPEG_Quality);
                        else
                            imwrite(im_noise, fullfile(save_LR_path, [imname '_' num2str(kernel_label, '%02d') num2str(noise_label, '%02d') num2str(jpeg_label, '%02d') file_type]));
                        end
                    end
                end
            end
        end
    end
end