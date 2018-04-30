function output = propagate(image, aperture_size)
%PROPAGATE Summary of this function goes here
%   Detailed explanation goes here

Nz = 100;	% Number of steps in the z direction <---### EDIT HERE ###

[Nx, Ny] = size(image);
% Physical dimension of the computation space. Physical values are denoted with 
% an underscore. The corresponding normalized value are written without underscore. 

Lx_ = 4e-5;		% width of the computation window [m]  <---### EDIT HERE ###
Ly_ = Lx_;		% height of the computation window [m] <---### EDIT HERE ###
Lz_ = 2e-4;     % propagation distance [m]             <---### EDIT HERE ###

n0_ = 1;	% linear refractive index (1.38 fore acetone) <---### EDIT HERE ###

lambda0_ = 250e-9; 	% free space wavelength [m] <---### EDIT HERE ###
delta_ = 1.0;       % normalization parameter (see documentation on SSF)
n2_ = 0;			% nonlinear coefficient [m2/W] (2.4e-19m2/W for acetone) <---### EDIT HERE ###
V = zeros(Ny, Nx);	% potential. This correspond to the refractive index difference profile in the transverse plane
					% for a homogeneous medium, V = 0. <---### EDIT HERE ###

% #########################################################
% SETUP THE SSF-BPM VARIABLES
% #########################################################

% Normally, you shouldn't need to edit this section

% Physical constants

mu0 = 4.0e-7 * pi;	% free space magnetic permeability [Vs/Am]
c0 = 2.99792458e+8;	% free space light speed [m/s]

epsilon0 = 1.0 / (mu0 * c0^2);	% free space permittivity [As/Vm]
eta0 = sqrt(mu0 / epsilon0);		% free space impedance [ohm]

% Derived parameters

n2_el = n0_ * n2_ / (2 * eta0);	% nonlinear refractive index [m2/V2]
k0_ = 2 * pi / lambda0_;			% free space wavenumber [m-1]
k_ = n0_ * k0_;						% medium wavenumber [m-1]
lambda_ = lambda0_ / n0_;			% medium wavelength [m]

% Normalization coefficients
% The equation can be normalized to a dimensionless form

% spatial normalization factor in the x-y plane
spatial_transverse_scale = 1/(k0_ * sqrt(2 * n0_ *  delta_));
% spatial normalization factor in the z direction
spatial_longitudinal_scale = 1/(delta_ * k0_);

scale_ratio = spatial_longitudinal_scale/spatial_transverse_scale; % = sqrt(2*n0_/delta_)
% normalization factor for the electric field
field_scale = sqrt(delta_ / n2_el);

% ************* Normalized parameters *************

Lx = Lx_ / spatial_transverse_scale;	% normalized model width
Ly = Ly_ / spatial_transverse_scale;	% normalized model height
Lz = Lz_ / spatial_longitudinal_scale;	% normalized propagation distance
k = 2*pi * spatial_transverse_scale / lambda_; % normalized light k-vector

% ************ Numeric model parameters ***********

dx_ = Lx_/Nx;		% normalized discretization step in x
dx = Lx/Nx;			% discretization step in x
x_ = dx_ * (-Nx/2:Nx/2-1);	% x dimension vector
x = dx * (-Nx/2:Nx/2-1);	% normalized x dimension vector
dkx = 2*pi/Lx;		% discretization in the spatial spectral domain along the y direction
kx = dkx * [0:Nx/2-1, -Nx/2:-1];	% spatial frequencies vector in the x direction
% Note that we swap the left and right part of this vector because the FFT algorithm puts the
% low frequencies in the edges of the transformed vector and the high frequencies in the middle.
% There is a fftshift function that perform this swap in MATLAB. Here we prefer to redefine k so
% that we don't need to call fftshift everytime we use fft.

% We do the same in the y and z direction
dy = Ly/Ny;
dy_ = Ly_/Ny;
y = dy * (-Ny/2:Ny/2-1)';
y_ = dy_ * (-Ny/2:Ny/2-1)';
dky = 2*pi/Ly;
ky = dky * [0:Ny/2-1 -Ny/2:-1]';

dz = Lz/Nz;
dz_ = Lz_/Nz;
z = dz * (1:Nz);
z_ = dz_ * (1:Nz);

% Here we create the spatial computation grid (physical and normalized)
[X_, Y_] = meshgrid(x_, y_);
[Xz_, Z_] = meshgrid(x_, z_);
[X, Y] = meshgrid(x, y);
[Xz, Z] = meshgrid(x, z);

% The same for the spatial frequencies domain
[Kx, Ky] = meshgrid(kx, ky);

K2 = Kx.^2 + Ky.^2; % Here we define some variable so that we don't need to compute them again and again
K = sqrt(K2);

%% Lens definition
% Aperture
radius = aperture_size*Lx_/2;
aperture = ones(size(image));
aperture((X_.^2 + Y_.^2) > radius.^2) = 0;

% Lens
focal = Lz_/4;
lens =  exp(-1i*pi/lambda_/focal.*(X_.^2+Y_.^2));
lens = lens;% * aperture;


%% Gaussian Beam
 beam_fwhm_ = Lx_/2;
 beam_scale_ = beam_fwhm_ / (2*sqrt(log(2)));
 P = 3; %Power of the field in [W]
 If_ = P/(pi*beam_scale_^2);
 A_ = sqrt(2 * eta0 * If_);	% Peak field amplitude of the input beam [V/m]
 A = A_ / field_scale;		% normalized peak field amplitude of the input beam

 gaussian_beam = A_ .* exp(-((X_/beam_scale_).^2 + (Y_/beam_scale_).^2).^5);

 u = image;

% Windowing function for absorbing boundary condition
window = exp(-(x_/(0.43*Lx_)).^20);
 
% #########################################################
% PROPAGATION ROUTINE (CORE OF THE STORY)
% #########################################################
%u = u.*window.*window';
u = ifft2(fft2(u) .* exp(-1i * K2 * Lz/4));     % First linear half step
u = u.*lens;
u = ifft2(fft2(u) .* exp(-1i * K2 * Lz/4));     % First linear half step
u = u.*aperture;
u = ifft2(fft2(u) .* exp(-1i * K2 * Lz/4));     % First linear half step
u = u.*lens;
u = ifft2(fft2(u) .* exp(-1i * K2 * Lz/4));     % First linear half step


% figure, imagesc(abs(u))

% Let's store the result in a variable with a more explicit name
output = u;

end

