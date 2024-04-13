function [MAP_result,eva,training_time] = train_twostep(XTrain,LTrain,XQuery,LQuery,param,anchor,h)

%% get the dimensions of features
n = param.trainset_size; % 16000   (because we delete the last 883 samples
dX = size(anchor,1);  % 1000 (1000*4096)
dY = param.label_size;  % 24

%% set the parameters
nbits = param.nbits; % length of the hash code

beta = param.beta;
theta = param.theta;
mu = param.mu;
delta = param.delta;
gamma = param.gamma;
lambda = param.lambda;

parameter = param.parameter; % 10

%% initialization

MAP_result=zeros(1,param.nchunks);  

% copy this to the class-wise hash code (leibie haxi ma)
Y = h'; % c*r

for chunki =1:param.nchunks
    fprintf('-----chunk----- %3d\n', chunki); 
    
    XTrain_new = XTrain{chunki,:};
    LTrain_new = LTrain{chunki,:};

    XQueryt = XQuery{chunki,:};
    LQueryt = LQuery{chunki,:};

    % RBF kernel mapping
    X = Kernelize(XTrain_new, anchor);
    X = X'; % d*n_t
    
    current_nt = size(X,2);
    
    
    
    B_new = sign(randn(nbits, current_nt));
    V_new = rand(nbits, current_nt);
    U_new = rand(dX,nbits);
    H_F = eye(current_nt);  % H_F = diag(1:current_nt); % initialzie H_F
    H_C = eye(dY); % H_C = diag(1:dY);
    D_new = eye(nbits); % D_new = diag(1:nbits);
%     E_new = rand(dX,nbits); % init or not init either is ok.
    W_new = rand(nbits,dX);
    
    fprintf('[%s][%d bits][%d chunk]\n',param.ds_name,nbits,chunki);  

    tic;
    
    if chunki == 1
    %%  low-level feature
        
        fprintf("*********lambda:{%f}**********\n",lambda);
        
        normytagA = ones(current_nt,1);% 588*1
      

        % norm simple YTrain, norm XTrain(after kernel)
        for i =1:current_nt %1:2000
            if norm(LTrain_new(i,:))~=0 % if current chunk's L's norm !=0
                normytagA(i,:)=norm(LTrain_new(i,:));% -d column vector
            end
            if norm(X(:,i))~=0 % if current chunk's L's norm !=0
                normX(:,i)=norm(X(:,i));% -d column vector (*1)
            end
        end

        % This is ||L||
        normytagA = repmat(normytagA,1,dY); % 
%         normX = repmat(normX,dX,1); % *1000

        % SA is G t arrow (Gt=Lt/||Lt||)
        SA_new = LTrain_new./normytagA; 

        
        for iter = 1:parameter
            
            % update H_F
            recons_error_F = vecnorm(X-U_new*V_new,2,1); % 1*2000
            for i=1:current_nt
                H_F(i,i) = (1+exp(-1/lambda))/(1+exp(recons_error_F(1,i)-(-1)/lambda));
            end
            
            % update H_C
            recons_error_C = vecnorm(LTrain_new-B_new'*Y,2,1); % 1*2000
            for i=1:dY
                H_C(i,i) = (1+exp(-1/lambda))/(1+exp(recons_error_C(1,i)-(-1)/lambda));
            end
            % update U
            U_new = (X*H_F*H_F'*V_new')*pinv(V_new*H_F*H_F'*V_new');
            
            % update V  eigenvalue decompositon(like DGH)
            Q = theta*U_new'*X*H_F*H_F'+nbits*(D_new*B_new*SA_new)*SA_new'+beta*D_new*B_new;
            
            Temp_QJQT = Q*(eye(current_nt)-(1/current_nt)*ones(1,current_nt)*ones(current_nt,1))*Q';
            [~,Lmd,QQ] = svd(Temp_QJQT);
            clear Temp_QJQT
            idx = (diag(Lmd)>1e-6);
            O = QQ(:,idx); O_ = orth(QQ(:,~idx));
            % The Qt of Temp and PP is opposite. 
            N = (Q'-(1/current_nt)*ones(current_nt,1)*(ones(1,current_nt)*Q')) *  (O / (sqrt(Lmd(idx,idx))));
            N_ = orth(randn(current_nt,nbits-length(find(idx==1))));
            V_new = sqrt(current_nt)*[O O_]*[N N_]';
            
            
            
            % update Y (using DCC)
            Gy = gamma*B_new*LTrain_new*H_C*H_C';
            Y=Y*H_C;
            for k=1:3
                for place=1:nbits
                    bit = 1:nbits;
                    bit(place) = [];
                    Y(place,:) = sign(Gy(place,:)' - Y(bit,:)'*B_new(bit,:)*B_new(place,:)')';  % y= sgn(c*1)';
                end
            end
            
            
            % update B (using DCC)
            Gb = D_new'*(V_new*SA_new)*SA_new' + beta*D_new'*V_new + gamma*Y*H_C*H_C'*LTrain_new';
            Y=Y*H_C;
            for k=1:3
                for place=1:nbits
                    bit = 1:nbits;
                    bit(place) = [];
                    B_new(place,:) = sign(Gb(place,:)' - B_new(bit,:)'*Y(bit,:)*Y(place,:)')';
                end
            end
            
        end
        
        % update E
        [a1, ~, a2] = svd(mu*X*X'*W_new','econ');
        E_new = a1*a2;
        % update W  (hash function)
        W_new = (1/(mu+1))*((D_new*B_new*X')+mu*E_new'*(X*X'))*pinv(X*X'+(delta/(mu+1))*eye(dX, dX));
        
        
        % save results
        H1_new = X*H_F*H_F'*V_new';
        H2_new = V_new*H_F*H_F'*V_new';
        H3_new = D_new*B_new*SA_new;
        H4_new = V_new*SA_new;
        H5_new = X*X';
        H6_new = D_new*B_new*X';

        HH{1,1} = H1_new;
        HH{2,1} = H2_new;
        HH{3,1} = H3_new;
        HH{4,1} = H4_new;
        HH{5,1} = H5_new;
        HH{6,1} = H6_new;
        BB{1,chunki} = B_new;
        
        
    end
    
    if chunki >= 2
        lambda = lambda*(chunki-1)*(1/nbits);
        fprintf("*********lambda:{%f}**********\n",lambda);
        
        normytagA = ones(current_nt,1);% 588*1
        

        % norm simple YTrain, norm XTrain(after kernel)
        for i =1:current_nt %1:2000
            if norm(LTrain_new(i,:))~=0 % if current chunk's L's norm !=0
                normytagA(i,:)=norm(LTrain_new(i,:));% -d column vector
            end
            if norm(X(:,i))~=0 % if current chunk's L's norm !=0
                normX(:,i)=norm(X(:,i));% -d column vector (*1)
            end
        end

        % This is ||L||
        normytagA = repmat(normytagA,1,dY); % 
%         normX = repmat(normX,dX,1); % *1000

        % SA is G t arrow (Gt=Lt/||Lt||)
        SA_new = LTrain_new./normytagA; 
        
        for iter = 1:parameter
            
            % update H_F
            recons_error_F = vecnorm(X-U_new*V_new,2,1); % 1*2000
            for i=1:current_nt
                H_F(i,i) = (1+exp(-1/lambda))/(1+exp(recons_error_F(1,i)-(-1)/lambda));
            end
            
            % update H_C
            recons_error_C = vecnorm(LTrain_new-B_new'*Y,2,1); % 1*2000
            for i=1:dY
                H_C(i,i) = (1+exp(-1/lambda))/(1+exp(recons_error_C(1,i)-(-1)/lambda));
            end
            % update U
            U_new = (X*H_F*H_F'*V_new'+HH{1,1})*pinv(V_new*H_F*H_F'*V_new'+HH{2,1});
            
            % update V  eigenvalue decompositon(like DGH)
            Q = theta*U_new'*X*H_F*H_F'+nbits*(D_new*B_new*SA_new+HH{3,1})*SA_new'+beta*D_new*B_new;
            
            Temp_QJQT = Q*(eye(current_nt)-(1/current_nt)*ones(1,current_nt)*ones(current_nt,1))*Q';
            [~,Lmd,QQ] = svd(Temp_QJQT);
            clear Temp_QJQT
            idx = (diag(Lmd)>1e-6);
            O = QQ(:,idx); O_ = orth(QQ(:,~idx));
            % The Qt of Temp and PP is opposite.
            N = (Q'-(1/current_nt)*ones(current_nt,1)*(ones(1,current_nt)*Q')) *  (O / (sqrt(Lmd(idx,idx))));
            N_ = orth(randn(current_nt,nbits-length(find(idx==1))));
            V_new = sqrt(current_nt)*[O O_]*[N N_]';
            
            
            
            % update Y (using DCC)
            Gy = gamma*B_new*LTrain_new*H_C*H_C';
            Y=Y*H_C;
            for k=1:3
                for place=1:nbits
                    bit = 1:nbits;
                    bit(place) = [];
                    Y(place,:) = sign(Gy(place,:)' - Y(bit,:)'*B_new(bit,:)*B_new(place,:)')';  % y= sgn(c*1)';
                end
            end
            
            
            % update B (using DCC)
            Gb = D_new'*(V_new*SA_new+HH{4,1})*SA_new' + beta*D_new'*V_new + gamma*Y*H_C*H_C'*LTrain_new';
            Y=Y*H_C;
            for k=1:3
                for place=1:nbits
                    bit = 1:nbits;
                    bit(place) = [];
                    B_new(place,:) = sign(Gb(place,:)' - B_new(bit,:)'*Y(bit,:)*Y(place,:)')';
                end
            end
            
        end
        
        % update E
        [a1, ~, a2] = svd(mu*(X*X'+HH{5,1})*W_new','econ');
        E_new = a1*a2;
        % update W  (hash function)
        
        W_new = (1/(mu+1))*((D_new*B_new*X'+HH{6,1})+mu*E_new'*(X*X'+HH{5,1}))*pinv(X*X'+HH{5,1}+(delta/(mu+1))*eye(dX, dX));
        % save results
        H1_new = X*H_F*H_F'*V_new';
        H2_new = V_new*H_F*H_F'*V_new';
        H3_new = D_new*B_new*SA_new;
        H4_new = V_new*SA_new;
        H5_new = X*X';
        H6_new = D_new*B_new*X';

        
        
        HH{1,1} = HH{1,1}+H1_new;
        HH{2,1} = HH{2,1}+H2_new;
        HH{3,1} = HH{3,1}+H3_new;
        HH{4,1} = HH{4,1}+H4_new;
        HH{5,1} = HH{5,1}+H5_new;
        HH{6,1} = HH{6,1}+H6_new;
        BB{1,chunki} = B_new;
        
        
    end
    training_time(1,chunki) = toc;
    
    fprintf('       : training ends, training time is %f,\nevaluation begins. \n',training_time(1,chunki));
    
    XKTest=Kernelize(XQueryt,anchor); % Da phi(XTest)
    XKTest = XKTest';

    BxTest = compactbit((W_new*XKTest)' >= 0);  % r*n_query

    B = cell2mat(BB(1,1:chunki));

    BxTrain = compactbit(B' >= 0);  % r*N_till_now
    
    DHamm = hammingDist(BxTest, BxTrain); % ntest * ntrain

    [~, orderH] = sort(DHamm, 2); % each row, from low to high
    
    % my mAP
    MAP  = mAP(orderH', cell2mat(LTrain(1:chunki,:)), LQueryt);
    [eva_info.precison, eva_info.recall] = precision_recall(orderH', cell2mat(LTrain(1:chunki,:)), LQueryt);
    fprintf('       : evaluation ends, MAP is %f\n',MAP);

    eva(1,chunki) = eva_info;
    MAP_result(1,chunki)=MAP;
    
end

