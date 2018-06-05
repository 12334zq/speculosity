% Parameters
origin = './../../data~/datasets/roughness2200/1024p/';
dest =  './../../data~/datasets/roughness2200/centred_patches_64p/';
classes = {'A2', 'A3', 'A4', 'A5', 'A6', 'A7',...
           'B2', 'B3', 'B4', 'B5', 'B6', 'B7'};
img_per_class = 2200;
shape = [64, 64];			% Shape of output image x,y
y_range = -shape(1)/2:shape(1)/2-1;
x_range = -shape(2)/2:shape(2)/2-1;
sampling_rate = 2;          % Sub sampling rate

% Data processing
for class_name = classes
	origindir = strcat(origin, class_name{1});
	destdir = strcat(dest, class_name{1});
	fprintf("Processing Folder = '%s'\n", class_name{1});
	tic
	for i = 0:(img_per_class - 1)
		orginfile = sprintf('%s/%d.bmp', origindir, i);
		destfile = sprintf('%s/%d.bmp', destdir, i);
		
		% Load image
		data = imread(orginfile);
		
		% Subsampling
        data = data(1:sampling_rate:end, 1:sampling_rate:end);
		
		% Cut patch
		data_thresh = data;
		data_thresh(data_thresh < 20) = 0;
		[y, x] = cg(data_thresh);
		data = data(y_range + round(y), x_range + round(x));
		
		% Save image
		imwrite(data, destfile);
	end
	toc
end

% Center of gavity of an array
function [y, x] = cg(X)
	S = size(X);
	y_flat = sum(X, 2);
	x_flat = sum(X, 1);
	total_sum = sum(x_flat);
	y = (0:(S(1)-1)) * y_flat / total_sum;
	x = (0:(S(2)-1)) * x_flat' / total_sum;
end