% This script is to transform the original dataset to bmp

% Parameters
dataset = './1';
col = 2;
row = 'A';
shape = [960, 1280];        % Shape of RAW image
sampling_rate = 1;          % Sub sampling rate
crop_shape = [128, 128] ;   % Size of output (sub sampled imaged is cropped)

% Derived Parameters
output_shape =  shape/2/sampling_rate;
crop_origin = (output_shape - crop_shape)/2;

% Data processing
while true
    origindir = sprintf("%s/raw/%c%d/", dataset, row, col);
    if ~exist(origindir, 'file')
        row = 'B';
        col = 2;
        continue
    else
        destdir = sprintf('%s/bmp/%c%d/', dataset, row, col);
        mkdir(destdir)
    end

    i = 0;
    while true
        filename = sprintf("%s%d.raw", origindir, i);
        try 
            data = openraw(filename, shape, 'uint8')';
            % Extracting red pixels
            data = data(1:2:end, 1:2:end);
            % Subsampling
            data = data(1:sampling_rate:end, 1:sampling_rate:end);
            % Cropping
            data = data(crop_origin(1)+1:crop_origin(1) + crop_shape(1),... 
                        crop_origin(2)+1:crop_origin(2) + crop_shape(2));
            imwrite(data/255, sprintf('%s%d.bmp', destdir, i));
        catch e
            i = 0;
            col = col + 1;
            break;
        end
            
        i = i + 1;
    end

end