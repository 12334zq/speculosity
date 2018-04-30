function create_dataset(train_size, val_size, test_size)
%UNTITLED Summary of this function goes here
%   Detailed explanation goes here

% Default parameter value
if nargin < 1
	train_size = 100;
end
if nargin < 2
	val_size = 10;
end
if nargin < 3
	test_size = 100;
end

img_size = 2^8;

create_subset('fwgn/test', img_size, test_size);
create_subset('fwgn/train', img_size, train_size);
create_subset('fwgn/val', img_size, val_size);

end

function create_subset(location, img_size, n_samples)
% CREATE_SUBSET Creates datapoint/label pairs.
%		CREATE_SUBSET(LOCATION, IMAGE_SIZE, N_SAMPLES) creates N_SAMPLES
%		datapoint/sample pares of size (IMAGE_SIZE x IMAGE_SIZE) and saves them
%		in the folder specified by LOCATION.
%
%		The data is genrated by propagating a image of size 2*IMAGE_SIZE thorugh
%		a finit apperture optical imaging system. The resulting output is
%		considered to be the datapoint and the initial image, the label.

type = "fwgn";

for i = 1:n_samples
	%% Refrence image (Label)
    img = random_pattern(2*img_size, type);
	% Crop image
    img_ = img(img_size/2+1:img_size*3/2, img_size/2+1:img_size*3/2);
	% Save image
    imwrite(img_, sprintf('%s/labels/%d.bmp', location, i));
	
	%% Input image (Data)
	% Propagate the input image
    img = propagate(img, 0.3);	
	% Flip the image back to have same orientation as the input image.
    img = abs(rot90(img,2));	
	% Crop image
    img_ = img(img_size/2+1:img_size*3/2, img_size/2+1:img_size*3/2);
	% Save image
    imwrite(img_, sprintf('%s/images/%d.bmp', location, i));
end
end