
% You should specify the location which BKSP is installed.
prefix = '/Users/shusaku/bksp';
openblas = '/usr/local/opt/openblas';

includes = ['-I' prefix '/include/ -I' openblas 'include'];
libs = ['-L' prefix '/lib/ -lbksp -L' openblas '/lib/ -lopenblas'];
list = dir('*.c');
comm = ['mex -c util.c ' includes ' -largeArrayDims'];
disp(comm);
eval(comm);
for i=1:length(list)
    if ~strcmp(list(i).name, 'util.c')
        comm = ['mex ' list(i).name ' util.c ' includes ' ' libs ' -largeArrayDims'];
        disp(comm);
        eval(comm);
    end
end