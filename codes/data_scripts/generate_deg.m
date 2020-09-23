function generate_mod_LR_bic()

% put training data here
input_path = '../../datasets/DIV2K_sub';
save_LR_path = '../../datasets/DIV2K_k2_noise30';

file_type = '.png';

kernelwidth = 2;
noiseSigma = 30;

kernel = single(fspecial('gaussian', 21, kernelwidth));

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

        if kernelwidth > 0
            im_blurry = imfilter(img, double(kernel), 'replicate');
        else
            im_blurry = img
        end
        noise = noiseSigma/255.*randn(size(img));
        im_noise = single(im_blurry + noise);
        im_final = im2uint8(im_noise);

        if exist('save_LR_path', 'var')
            imwrite(im_final, fullfile(save_LR_path, [imname file_type]));
        end
    end
end
end