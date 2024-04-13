clear;
addpath(genpath('./'));

nbits_set=[16 32 64 96 128];

%% load dataset
fprintf('loading dataset...\n');

param.ds_name = 'CIFAR10';
% param.ds_name = 'MIRFLICKR';
% param.ds_name = 'NUS-WIDE';

[param,anchor,XTrain,LTrain,XQuery,LQuery] = load_dataset(param);

set = param.ds_name;
trainset_size = param.trainset_size;
label_size = param.label_size;
%% initialization
fprintf('initializing...\n')
if strcmp(set,'MIRFLICKR')
    % MIR [already make sure params]
    param.gamma = 0.001; 
    param.beta = 10;
    param.delta = 0.001;
    param.theta = 10;
    param.mu = 0.001; 
    param.lambda = 1000;
    
elseif strcmp(set,'NUS-WIDE')
    % NUSWIDE [already majke sure params]
    param.gamma = 100;
    param.beta = 10;
    param.delta = 0.001;
    param.theta = 0.1; 
    param.mu = 0.001;
    param.lambda = 1000;
    
elseif strcmp(set,'CIFAR10')
    % CIFAR10 [already make sure params]
    param.gamma = 100;
    param.beta = 10;  
    param.delta = 0.001; 
    param.theta = 0.1; 
    param.mu = 0.1;
    param.lambda = 1000;
    
end

param.datasets = set;

param.parameter = 10;


%% model training
for bit=1:length(nbits_set)
    
    nbits=nbits_set(bit);
    param.nbits=nbits;
    
    % randomly generate hadamard codebook
    if strcmp(param.datasets,'MIRFLICKR')
        h = hadamard(512); % 404tags/ 24label
        h = h(randperm(label_size),randperm(nbits)); % 404*nbits
    elseif strcmp(param.datasets,'NUS-WIDE')
        h = hadamard(8192); % 5000
        h = h(randperm(label_size),randperm(nbits)); % 5000*nbits
    elseif strcmp(param.datasets,'CIFAR10')
        h = hadamard(256); % 10
        h = h(randperm(label_size),randperm(nbits)); % 5000*nbits

    end

[ MAP(bit,:),eva(bit,:),training_time(bit,:)] = train_twostep(XTrain,LTrain,XQuery,LQuery,param,anchor,h);

end 


