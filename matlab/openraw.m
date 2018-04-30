function data = openraw(filename, shape, type)

fid = fopen(filename, 'r');

if fid == -1
  error('Cannot open file: %s', FileName);
end

data = fread(fid, prod(shape) , type);
fclose(fid);
data = reshape(data, flip(shape));

return

