function generate_mod_LR_bic()

% put training data here
input_path = '../../datasets/DIV2K_sub';
save_LR_path = '../../datasets/DIV2K_beta_0510_2group_k04_noise0050_41x51';

file_type = '.png';

% kernel
kernel_label_list = 0:1:40;
kernel_list = kernel_label_list/10.;
kernel_length = length(kernel_list)

% noise
noise_list = 0:1:50;
noise_label_list = noise_list;
noise_length = length(noise_label_list)

beta_a = 0.5;
beta_b = 1.0;

if exist('save_mod_path', 'var')
    if exist(save_mod_path, 'dir')
        disp(['It will cover ', save_mod_path]);
    else
        mkdir(save_mod_path);
    end
end
if exist('save_LR_path', 'var')
    if exist(save_LR_path, 'dir')
        disp(['It will cover ', save_LR_path]);
    else
        mkdir(save_LR_path);
    end
end

randn('seed', 0);

idx = 0;
filepaths = dir(fullfile(input_path,'*.*'));
for i = 1 : length(filepaths)
    % randn('seed', 0);
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

        group_index = randi(2);

        if group_index == 1
            % one degradation
            type_index = randi(2);
            if type_index == 1

                % beta
                kernel_label = round(betarnd(beta_a, beta_b)*40.);

                kernelwidth = kernel_label/10.;

                if kernel_label > 0
                    kernel = single(fspecial('gaussian', 21, kernelwidth));
                    blurry_img = imfilter(img, double(kernel), 'replicate');
                else
                    blurry_img = img;
                end

                im_noise = blurry_img;

                noise_label = 0;

            elseif type_index == 2

                noise_label = round(betarnd(beta_a, beta_b)*50.);
                noiseSigma = noise_label;

                noise = noiseSigma/255.*randn(size(img));
                im_noise = single(img + noise);
                im_noise = im2uint8(im_noise);

                kernel_label = 0;
            end

        elseif group_index == 2

            kernel_label = round(betarnd(beta_a, beta_b)*40.);
            kernelwidth = kernel_label/10.;

            if kernel_label > 0
                kernel = single(fspecial('gaussian', 21, kernelwidth));
                blurry_img = imfilter(img, double(kernel), 'replicate');
            else
                blurry_img = img;
            end

            noise_label = round(betarnd(beta_a, beta_b)*50.);
            noiseSigma = noise_label;

            noise = noiseSigma/255.*randn(size(blurry_img));
            im_noise = single(blurry_img + noise);
            im_noise = im2uint8(im_noise);
        end

        if exist('save_LR_path', 'var')
            imwrite(im_noise, fullfile(save_LR_path, [imname '_' num2str(kernel_label, '%02d') num2str(noise_label, '%02d') file_type]));
        end

    end
end
end