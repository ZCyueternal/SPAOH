function [train_param,anchor,XTrain,LTrain,XQuery,LQuery] = load_dataset(train_param)
    fprintf(['-------load dataset------', '\n']);
    
    if strcmp(train_param.ds_name, 'MIRFLICKR')
        
        load('../Datasets/MIRFLICKR.mat');
       
        train_param.image_feature_size = 512;  % 4096
%         train_param.text_feature_size = 1386;
        train_param.trainset_size = size(I_tr, 1);
        train_param.label_size = size(L_tr, 2);
        
        X = [I_tr; I_te];
        L = [L_tr; L_te];
        
        anchor = I_tr(randsample(2000,1000),:); %% random select 1000 sample from XTrain (1000*512)
        
        R = randperm(size(L,1));
        queryInds = R(1:2000);
        sampleInds = R(2001:end);
        
        train_param.nq = 200; 
        train_param.n1 = 100;
        train_param.chunk = 2000;
        train_param.nmax = 1000;
        
        train_param.nchunks = floor(length(sampleInds)/train_param.chunk);
        train_param.chunksize = cell(train_param.nchunks,1);
        train_param.test_chunksize = cell(train_param.nchunks,1);
        
        XTrain = cell(train_param.nchunks,1);
        LTrain = cell(train_param.nchunks,1);

        XQuery = cell(train_param.nchunks,1);
        LQuery = cell(train_param.nchunks,1);
        
        for subi = 1:train_param.nchunks-1
            XTrain{subi,1} = X(sampleInds(train_param.chunk*(subi-1)+1:train_param.chunk*subi),:);
            LTrain{subi,1} = L(sampleInds(train_param.chunk*(subi-1)+1:train_param.chunk*subi),:);
            [train_param.chunksize{subi, 1},~] = size(XTrain{subi,1});

            XQuery{subi,1} = X(queryInds, :);
            LQuery{subi,1} = L(queryInds, :);
            [train_param.test_chunksize{subi, 1},~] = size(XQuery{subi,1});

        end

        XTrain{train_param.nchunks,1} = X(sampleInds(train_param.chunk*subi+1:end),:);
        LTrain{train_param.nchunks,1} = L(sampleInds(train_param.chunk*subi+1:end),:);
        [train_param.chunksize{train_param.nchunks, 1},~] = size(XTrain{train_param.nchunks,1});

        XQuery{train_param.nchunks,1} = X(queryInds, :);
        LQuery{train_param.nchunks,1} = L(queryInds, :);
        
        [train_param.test_chunksize{train_param.nchunks, 1},~] = size(XQuery{train_param.nchunks,1});


    elseif strcmp(train_param.ds_name, 'NUS-WIDE')   
        
        load('../Datasets/NUSWIDE10.mat');
        train_param.image_feature_size=500;
        
        train_param.trainset_size = size(I_tr, 1);
        train_param.label_size = size(L_tr, 2);
        
        X = [I_tr; I_te];
        L = [L_tr; L_te];
%         X = X(1:40000,:);
%         L = L(1:40000,:);
        
        anchor = I_tr(randsample(2000,1000),:); %% random select 1000 sample from XTrain (1000*500)
        
        R = randperm(size(L,1));
        queryInds = R(1:2000);
        sampleInds = R(2001:42000);
        
        
        train_param.nq = 400;
        train_param.n1 = 100;
        train_param.chunk = 10000;
        train_param.nmax = 1000;
        
        
        train_param.nchunks = floor(length(sampleInds)/train_param.chunk);
        train_param.chunksize = cell(train_param.nchunks,1);
        train_param.test_chunksize = cell(train_param.nchunks,1);
        
        XTrain = cell(train_param.nchunks,1);
        LTrain = cell(train_param.nchunks,1);

        XQuery = cell(train_param.nchunks,1);
        LQuery = cell(train_param.nchunks,1);
        
        for subi = 1:train_param.nchunks-1
            XTrain{subi,1} = X(sampleInds(train_param.chunk*(subi-1)+1:train_param.chunk*subi),:);
            LTrain{subi,1} = L(sampleInds(train_param.chunk*(subi-1)+1:train_param.chunk*subi),:);
            [train_param.chunksize{subi, 1},~] = size(XTrain{subi,1});

            XQuery{subi,1} = X(queryInds, :);
            LQuery{subi,1} = L(queryInds, :);
            [train_param.test_chunksize{subi, 1},~] = size(XQuery{subi,1});

        end

        XTrain{train_param.nchunks,1} = X(sampleInds(train_param.chunk*subi+1:end),:);
        LTrain{train_param.nchunks,1} = L(sampleInds(train_param.chunk*subi+1:end),:);
        [train_param.chunksize{train_param.nchunks, 1},~] = size(XTrain{train_param.nchunks,1});

        XQuery{train_param.nchunks,1} = X(queryInds, :);
        LQuery{train_param.nchunks,1} = L(queryInds, :);
        
        [train_param.test_chunksize{train_param.nchunks, 1},~] = size(XQuery{train_param.nchunks,1});
        
        clear X L subi queryInds sampleInds R
    
    elseif strcmp(train_param.ds_name, 'CIFAR10')   
        
        load('../Datasets/cifar10-cut-follow-FOH.mat');
        L_tr = L_tr_onehot;
        L_te = L_te_onehot;
        train_param.image_feature_size=4096;
        
        train_param.trainset_size = size(I_tr, 1);
        train_param.label_size = size(L_tr, 2);
        
        X = [I_tr; I_te];
        L = [L_tr; L_te];
        
        anchor = I_tr(randsample(2000,1000),:); %% random select 1000 sample from XTrain (1000*4096)
        
        R = randperm(size(L,1));
        queryInds = R(1:1000);
        sampleInds = R(1001:end);
        
        
        train_param.nq = 200;
        train_param.n1 = 100;
        train_param.chunk = 2000;
        train_param.nmax = 1000;
        
        
        train_param.nchunks = floor(length(sampleInds)/train_param.chunk);
        train_param.chunksize = cell(train_param.nchunks,1);
        train_param.test_chunksize = cell(train_param.nchunks,1);
        
        XTrain = cell(train_param.nchunks,1);
        LTrain = cell(train_param.nchunks,1);

        XQuery = cell(train_param.nchunks,1);
        LQuery = cell(train_param.nchunks,1);
        
        for subi = 1:train_param.nchunks-1
            XTrain{subi,1} = X(sampleInds(train_param.chunk*(subi-1)+1:train_param.chunk*subi),:);
            LTrain{subi,1} = L(sampleInds(train_param.chunk*(subi-1)+1:train_param.chunk*subi),:);
            [train_param.chunksize{subi, 1},~] = size(XTrain{subi,1});

            XQuery{subi,1} = X(queryInds, :);
            LQuery{subi,1} = L(queryInds, :);
            [train_param.test_chunksize{subi, 1},~] = size(XQuery{subi,1});

        end

        XTrain{train_param.nchunks,1} = X(sampleInds(train_param.chunk*subi+1:end),:);
        LTrain{train_param.nchunks,1} = L(sampleInds(train_param.chunk*subi+1:end),:);
        [train_param.chunksize{train_param.nchunks, 1},~] = size(XTrain{train_param.nchunks,1});

        XQuery{train_param.nchunks,1} = X(queryInds, :);
        LQuery{train_param.nchunks,1} = L(queryInds, :);
        
        [train_param.test_chunksize{train_param.nchunks, 1},~] = size(XQuery{train_param.nchunks,1});
        
        clear X L subi queryInds sampleInds R
        
        
        
    
    end
    
    fprintf('-------load data finished-------\n');
    clear I_tr I_te L_tr L_te
end

