clear all; close all; clc

addpath('C:\Users\Dominik\Documents\MATLAB\Low rank approximations\rank_adaptive_integrator_for_TTN')

% number particles
L = 8;

% parameters of the model
Omega = 0; % 0.4;
Delta = 0; % -2;
gamma = 0; % 1;
alpha = 10;
c_alpha=sum((1:1:L).^(-alpha));
V = 1; % 2/c_alpha;
r = [4 4 4 4 4]; % bottom layer: between 2 and 4

% bond dimension operator
r_op_max_save = [2 3 4 6 8 10 12 14 16 18 20 22];
r_op_min = 2;

% initialisations
sx=[0,1;1,0];  %% Pauli Matrix x
sy=[0,-1i;1i,0]; %% Pauli Matrix y
sz=[1,0;0,-1]; %% Pauli Matrix z
n=[1,0;0,0];  %% Projector onto the excited state Pu=(sz+id)/2;
id=[1,0;0,1];  %% Identity for the single spin
J = [0,0;1,0];

% % % create initial data
[X_TTN,tau_TTN] = init_diss_all_dim_diff_rank(r,2,L);
[X_TT,tau_TT] = init_spin_all_dim_same_rank_TT(L);
[X_tree4,tau_tree4] = init_spin_all_dim_diff_rank_tree4_d8(L); % only for d=16
% % % 
% % test Tucker
% X_TTN = cell(1,L+2);
% for ii=1:L
%     X_TTN{ii} = [1 1;0 1];
% end
% X_TTN{end-1} = 1;
% tmp = 2*ones(1,L);
% C = zeros(tmp);
% C(1) = 1;
% s = size(C);
% X_TTN{end} = tensor(C,[s 1]);
% tau_bin = cell(1,4);

% create cell array for Hamiltonian
% B = linearisation_long_range(L,J,sx,n,V,Delta,Omega,gamma,alpha);
% test
B = linearisation_long_range_test(L,J,sx,n,V,Delta,Omega,gamma,alpha);
V_mat = test_V_matrix(L,J,sx,n,V,Delta,Omega,gamma,alpha);

[~,S,~] = svd(V_mat)

% make operator 
A_TTN = make_operator(X_TTN,B,tau_TTN,4*ones(L,1));
A_TTN = rounding(A_TTN,tau_TTN);

A_TT = make_operator_TT(X_TT,B,tau_TT,4*ones(L,1));
A_TT = rounding(A_TT,tau_TT);

A_tree4 = make_operator(X_tree4,B,tau_TTN,4*ones(L,1));
A_tree4 = rounding(A_tree4,tau_tree4);

err_TTN = [];
err_TT = [];

free_TTN = [];
free_TT = [];

for ii=1:length(r_op_max_save)
    r_op_max = r_op_max_save(ii);
    
    % TTN
    A_TTN_trunc = truncate(A_TTN,10^-12,r_op_max,r_op_min);
    tmp = A_TTN_trunc;
    tmp{end} = -tmp{end};
    E = Add_TTN(A_TTN,tmp,tau_TTN);
    err_TTN(ii) = sqrt(abs(Mat0Mat0(E,E)));
    free_TTN(ii) = count_free_parameter(A_TTN_trunc);
    
    % TT/MPS
    A_TT_trunc = truncate(A_TT,10^-12,r_op_max,r_op_min);
    tmp = A_TT_trunc;
    tmp{end} = -tmp{end};
    E = Add_TTN(A_TT,tmp,tau_TT);
    err_TT(ii) = sqrt(abs(Mat0Mat0(E,E)));
    free_TT(ii) = count_free_parameter(A_TT_trunc);
    
end

figure(1)
semilogy(r_op_max_save,err_TTN,'Linewidth',2)
hold on 
semilogy(r_op_max_save,err_TT,'Linewidth',2)
legend('TTN','MPS','Fontsize',15,'location','southeast')

figure(2)
semilogy(free_TTN,err_TTN,'Linewidth',2)
xlabel('Free parameters')
ylabel('Error of truncated operator')
hold on 
semilogy(free_TT,err_TT,'Linewidth',2)
xlabel('Free parameters')
ylabel('Error of truncated operator')
legend('TTN','MPS','Fontsize',15,'location','southeast')

figure(3)
subplot(1,2,1), semilogy(r_op_max_save,err_TTN,'Linewidth',2)
title('Binary tree of minimal height')
xlabel('Max. rank of truncated operator')
ylabel('Error of truncated operator')
hold on 
subplot(1,2,2), semilogy(r_op_max_save,err_TT,'Linewidth',2)
title('Binary tree of maximal height')
xlabel('Max. rank of truncated operator')
ylabel('Error of truncated operator')



