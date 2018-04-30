function image = random_pattern(image_size, type, varargin)
%CREATE_PATTERN Creates an image with a set number of features.
%       CREATE_PATTERN(IMAGE_SIZE) creates an image of dimention (image_size x
%       image_size) with 10 uniformaly spread gaussian distributions with sigma
%       = 1.
%
%       CREATE_PATTERN(IMAGE_SIZE, TYPE) cerate an image with distributions of type
%       TYPE. Either "gaussian", "bessel", "wgn" for gaussian noise or "fwgn"
%       for spacially filtered white gaussian noise. Gaussian and bessel
%       distributions use a random orientation size.
%
%       CREATE_PATTERN(IMAGE_SIZE, TYPE, N_FEATURES) only for type = "gaussian" or 
%       "bessel". Number of feature to include in the image, defaults to 10.
%
%       CREATE_PATTERN(IMAGE_SIZE, TYPE, N_FEATURES, SIGMA) only for type = 
%       "gaussian" or "bessel". SIGMA is a 2x2 symetric matrix specifing the
%       covariance of the pdf.
%
%       CREATE_PATTERN(IMAGE_SIZE, TYPE, THRES) only for type = "fwgn". THRES is the
%       spacial frequency threshold for filtering of the white gaussian noise.

%% Default Parameters
intensity_range = [0.2 1];
size_range = [50 50];
size_ratio_range =  [1 1];
n_features = 10;
thresh = 22;

if nargin == 3
    n_features = varargin{1};
end

%% Creating image
[X1, X2] = meshgrid(0:image_size-1, 0:image_size-1);
grid = [X1(:) X2(:)];
image = zeros(image_size);

% If feature based pattern
if strcmp(type, "gaussian") || strcmp(type, "bessel") || strcmp(type, "rings")
	% Feature centroides randomization
    mu = scale(rand(n_features,2), [image_size/4 image_size*3/4-1]);
	% Feature maximum intensity randomization
    intensity = scale(rand(n_features,1), intensity_range);
	% Feature size (scale) randomization
    size = scale(rand(n_features,1), size_range);
    
	
	if strcmp(type, "gaussian")
		%% Generating gaussian features
        for i = 1:n_features
			% Creating randomized covariance matrix
			P = orth(randn(2));
			D = diag([1, scale(rand(), size_ratio_range)]);
			A = P*D*P';
			
			% Creating gaussian feature
			F = mvnpdf([X1(:) X2(:)], mu(i,:), A.*size(i));
			F = F/max(F(:)) * intensity(i); 
			F = reshape(F, image_size, image_size);
			image = image + F; 
        end
	elseif strcmp(type, "bessel")
		%% Generating bessel features
		% Random orientations
		theta = rand(n_features, 1)*2*pi;
		for i = 1:n_features
			% Cordonate rotation and scaling as to randomize orientation and
			% scale
			R = [cos(theta(i)), -sin(theta(i))
                 sin(theta(i)),  cos(theta(i))];
			S = diag([1, scale(rand(), size_ratio_range)]);
			S = S/size(i)*20;
			grid_ = grid - mu(i,:);
			grid_ = (S*R*grid_')';

			% Generating 
			dist = vecnorm(grid_, 2, 2);
			F = besselj(1, dist);
			F = reshape(F, image_size, image_size);
			image = image + F; 
		end
    elseif strcmp(type, "rings")
       %% Generating rings features
		% Random orientations
		theta = rand(n_features, 1)*2*pi;
		for i = 1:n_features
			% Cordonate rotation and scaling as to randomize orientation and
			% scale
			R = [cos(theta(i)), -sin(theta(i))
                 sin(theta(i)),  cos(theta(i))];
			S = diag([1, scale(rand(), size_ratio_range)]);
			S = S/size(i)*10;
			grid_ = grid - mu(i,:);
			grid_ = (S*R*grid_')';

			% Generating 
			dist = vecnorm(grid_, 2, 2);
            F  = ring(dist);
			F = reshape(F, image_size, image_size);
			image = image + F; 
		end
    end
else
 	if strcmp(type, "wgn")
		image = normrnd(0, 1, image_size, image_size);
	elseif strcmp(type, "fwgn")
		image = normrnd(0, 1, image_size, image_size);
		image_fft = fftshift(fft2(image));
		% Filtering
		image_fft((X1-image_size/2).^2 + (X2-image_size/2).^2 >thresh^2) = 0 ;
		image = real(ifft2(fftshift(image_fft)));
    end
end

% Min Max Normalization
image = normalize(image);

% Flat top gaussian for smoothing the sides to 0
x_ = linspace(-image_size/2, image_size/2, image_size);
window = exp(-(x_/(0.30*image_size)).^8);
image = image.*window.*window';

% figure, imagesc(image)

end

function y = normalize(x)
% NORMALIZE Normalizes the vector X between 0 and 1
	y = (x - min(x(:))) / (max(x(:)) - min(x(:)));
end

function y = scale(x, range)
% SCALE Scales the vector X between range(1) and range(2), the input is expected
% to be scaled between 0 and 1
	y = x*(range(2) - range(1)) + range(1);
end

function y = ring(x)
    y = zeros(size(x));
    y(x < 2*pi) = 1 - cos(x(x < 2*pi));
end
	
