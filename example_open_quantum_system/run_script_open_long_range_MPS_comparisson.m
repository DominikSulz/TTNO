clear all; close all; clc

addpath('C:\Users\Dominik\Documents\MATLAB\Low rank approximations\TTNO')
addpath('C:\Users\Dominik\Documents\MATLAB\Low rank approximations\TTNO\HSS_and_no_structure_case')
addpath('C:\Users\Dominik\Documents\MATLAB\Matlab toolboxes\hm-toolbox-master')
addpath('C:\Users\Dominik\Documents\MATLAB\Matlab toolboxes\tensor_toolbox-master')

addpath('C:\Users\Dominik\Documents\MATLAB\Low rank approximations\rank_adaptive_integrator_for_TTN')

% the code works now and gives good results. 2 things are to be discussed:
% 1) Depending on the parameters, the Frobeniusnorm of the TTNO get super
% big (10^80) such that svd etc. have trouble and potentially crash. 
% 2) Probably linked to 1). The error in scaledFrobeniusnorm is not of maschine
% precision anymore when taking many particles. Probably because the norm
% in total is to big.

% parameters of the model
Omega = 0.4;
Delta = -2;
gamma = 1;
alpha = 1;

r_vec = [10 10 10 10 10 10 10 6 4]; 

% initialisations
sx=[0,1;1,0];     %% Pauli Matrix x
sy=[0,-1i;1i,0];  %% Pauli Matrix y
sz=[1,0;0,-1];    %% Pauli Matrix z
n=[1,0;0,0];      %% Projector onto the excited state Pu=(sz+id)/2;
id=[1,0;0,1];     %% Identity for the single spin
J = [0,0;1,0];

rk = [1 2 3 4];

for kk=1:length(rk)
    d = 2^rk(kk);           % number of particles
    l = log(d)/log(2);      % number of layers
    
    c_alpha=sum((1:1:d).^(-alpha));
    nu = 2/c_alpha;
    
    [X,tau] = init_diss_all_dim_diff_rank(r_vec,2,d); % initial data - binary balanced tree
    [X_TT,tau_TT] = init_spin_all_dim_diff_rank_TT(d); % initial data - MPS
    
    %% interaction matrix - single particle
    A_single = cell(1,d);
    for ii=1:d
        A_single{ii} = Omega*(-1i*kron(sx,id) + 1i*kron(id,sx.')) ...
                     + Delta*(-1i*kron(n,id) + 1i*kron(id,n.')) ...
                     + gamma*(kron(J,conj(J)) - 0.5*kron(J'*J,id) - 0.5*kron(id,(J'*J).'));
    end
    V_single = eye(d,d);
    
    %% interaction matrix - long-range interactions
    V_int1 = zeros(d,d);
    A_int1 = cell(1,d);
    for ii=1:d
        for jj=1:d
            if ii ~= jj
                V_int1(ii,jj) = -1i*nu * 1/abs(ii-jj)^alpha; % -1i*(nu/2) * 1/abs(ii-jj)^alpha;
            end
        end
        A_int1{ii} = kron(n,id);
    end
    
    V_int2 = zeros(d,d);
    A_int2 = cell(1,d);
    for ii=1:d
        for jj=1:d
            if ii ~= jj
                V_int2(ii,jj) = 1i*nu * 1/abs(ii-jj)^alpha; % 1i*(nu/2) * 1/abs(ii-jj)^alpha;
            end
        end
        A_int2{ii} = kron(id,n.');
    end
    % *0.5 is away, as for the TTNO we only use the upper triangluar part of V,
    % while in the direct construction we use the full matrix V
    % --> also change that in the draft!
    
    %% HSS
    H_single = hss(V_single,'cluster',1:d);
    H_single = adjust_H(H_single);
    
    H_int1 = hss(V_int1,'cluster',1:d);
    H_int1 = adjust_H(H_int1);
    H_int2 = hss(V_int2,'cluster',1:d);
    H_int2 = adjust_H(H_int2);
    
    %% construction of TTNO with help of HSS
    TTNO_single = TTNO_HSS_construct_single(H_single,A_single,l,l,4*ones(1,d),1:d);
    TTNO_int1 = TTNO_HSS_construct(H_int1,A_int1,l,l,4*ones(1,d),1:d);
    TTNO_int2 = TTNO_HSS_construct(H_int2,A_int2,l,l,4*ones(1,d),1:d);
    
    TTNO = Add_TTN(TTNO_single,TTNO_int1,tau);
    TTNO = rounding(TTNO,tau);
    TTNO = Add_TTN(TTNO,TTNO_int2,tau);
    TTNO = rounding(TTNO,tau);
    
    %% TTNO in TT representation
    % with hss code
%     z = sparse([]);
% %     % old 
%     for jj=1:d-1
%         tmp = jj*ones(1,2^(d-jj-1));
%         z = [z tmp];
%     end
%     z(end+1) = d;
    
    z = sparse([]);
    for jj=1:d-1
        if (2^(d-jj-1) - 2) > 0
            z = [z jj sparse(1,2^(d-jj-1) - 2) jj];
        elseif (2^(d-jj-1) - 2) == 0
            z = [z jj jj];
        else
            z = [z jj];
        end
    end
    z = [z d];
    
    H_single_TT = hss(V_single,'cluster',z);   %  example: H = hss(V_int1,'cluster',[1 1 1 1 1 1 1 1 2 2 2 2 3 3 4 64]);
    H_int1_TT = hss(V_int1,'cluster',z);
    H_int2_TT = hss(V_int2,'cluster',z);
    
    TTNO_TT_single = TTNO_HSS_construct_single_TT(H_single_TT,A_single,l,l,4*ones(1,d),1:d);
    TTNO_TT_int1 = TTNO_HSS_construct_TT(H_int1_TT,A_int1,l,l,4*ones(1,d),1:d);
    TTNO_TT_int2 = TTNO_HSS_construct_TT(H_int2_TT,A_int2,l,l,4*ones(1,d),1:d);
    
    TTNO_TT = Add_TTN(TTNO_TT_single,TTNO_TT_int1,tau_TT);
%     TTNO_TT = rounding(TTNO_TT,tau_TT);
    TTNO_TT = Add_TTN(TTNO_TT,TTNO_TT_int2,tau_TT);
    TTNO_TT = rounding(TTNO_TT,tau_TT);
%     TTNO_TT = truncate(TTNO_TT,10^-10,100,2);
    
%     % with unstructured code
%     TTNO_TT_single = TTNO_no_structure_abitrary_tree(A_single,V_single,X_TT,l,l,4*ones(d,1),1:d);
%     TTNO_TT_int1 = TTNO_no_structure_abitrary_tree(A_int1,V_int1,X_TT,l,l,4*ones(d,1),1:d);
%     TTNO_TT_int2 = TTNO_no_structure_abitrary_tree(A_int2,V_int2,X_TT,l,l,4*ones(d,1),1:d);
%     
%     TTNO_TT = Add_TTN(TTNO_TT_single,TTNO_TT_int1,tau_TT);
% %     TTNO_TT = rounding(TTNO_TT,tau_TT);
%     TTNO_TT = Add_TTN(TTNO_TT,TTNO_TT_int2,tau_TT);
%     TTNO_TT = rounding(TTNO_TT,tau_TT);
% %     TTNO_TT = truncate(TTNO_TT,10^-10,100,2);
    
    %% max ranks    
    max_rk(kk) = max_rank(TTNO);
    max_rk_TT(kk) = max_rank(TTNO_TT);
    
    ex_rk(kk) = hssrank(H_int1) + 2 + hssrank(H_int2) + 2 + 1;
    ex_rk_TT(kk) = hssrank(H_int1_TT) + 2 + hssrank(H_int2_TT) + 2 + 1;
    
    kk

    % error check of MPS - ToDo
    
    
end

% plot
plot(2.^rk,max_rk,'Linewidth',2)
hold on
plot(2.^rk,max_rk_TT,'--','Linewidth',2)
plot(2.^rk,ex_rk,'-.','Linewidth',2)
plot(2.^rk,ex_rk_TT,':','Linewidth',2)
xlabel('Number of particles','Fontsize',12)
legend('Maximal rank of the TTNO binary tree','Maximal rank of the TTNO TT',...
'Expected rank binary tree','Expected rank TT','Fontsize',12)





% old code
% % test with own hss code
% %     z = [];
% % %     % old
% %     for jj=1:d
% %         tmp = [jj jj];
% %         z = [z tmp];
% %     end
% z = 1:d;
% 
% H_single_TT2 = hss_TT(V_single,'cluster',z);
% H_int1_TT2 = hss_TT(V_int1,'cluster',z);
% H_int2_TT2 = hss_TT(V_int2,'cluster',z);
% 
% TTNO_TT_single2 = TTNO_HSS_construct_single_TT(H_single_TT2,A_single,l,l,4*ones(1,d),1:d);
% TTNO_TT_int12 = TTNO_HSS_construct_TT(H_int1_TT2,A_int1,l,l,4*ones(1,d),1:d);
% TTNO_TT_int22 = TTNO_HSS_construct_TT(H_int2_TT2,A_int2,l,l,4*ones(1,d),1:d);
% 
% TTNO_TT2 = Add_TTN(TTNO_TT_single2,TTNO_TT_int12,tau_TT);
% %     TTNO_TT = rounding(TTNO_TT,tau_TT);
% TTNO_TT2 = Add_TTN(TTNO_TT2,TTNO_TT_int22,tau_TT);
% TTNO_TT2 = rounding(TTNO_TT2,tau_TT);
% %     TTNO_TT = truncate(TTNO_TT,10^-10,100,2);