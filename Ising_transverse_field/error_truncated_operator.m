% Error operator check
clear all; close all; clc
addpath('C:\Users\Dominik\Documents\MATLAB\Low rank approximations\rank_adaptive_integrator_for_TTN')

% number particles
L = 32;
% parameters of the model
Omega = 1;
% time step size and final time
T_end = 0.1;
dt = 0.01;
% rank of initial data at bottom layer
r = 2;
% bond dimension operator
r_op_max = 100;
r_op_min = 2;

%%% time-step
Iter=T_end/dt;

% initialisations
sx=[0,1;1,0];  %% Pauli Matrix x
sy=[0,-1i;1i,0]; %% Pauli Matrix y
sz=[1,0;0,-1]; %% Pauli Matrix z

% create initial data binary tree
[X_bin,tau_bin] = init_spin_all_dim_same_rank(r,2,L);

% create initial data tensor train
[X_TT,tau_TT] = init_spin_all_dim_diff_rank_TT(L);

% create cell array for Hamiltonian
B = linearisation_Ising(Omega*sx,sz,L);

% make operator of Ising model in binary tree representation
A = make_operator(X_bin,B,tau_bin,2*ones(L,1));
A{end} = -A{end};
A = rounding(A,tau_bin);
% A = truncate(A,10^-8,r_op_max,r_op_min);

app = apply_operator_nonglobal(X_bin,A,L);

% make operator of Ising model in tensor train representation
A_TT = make_operator_TT(X_TT,B,tau_TT,2*ones(L,1));
A_TT{end} = -A_TT{end};
A_TT = rounding(A_TT,tau_TT);

app_TT = apply_operator_nonglobal_TT(X_TT,A_TT,L);

err = [];
err_TT = [];
err_app = [];
err_app_TT = [];
r = [2 3 4 6 8 12 16];

for ii=1:length(r)
    % binary tree
    r_op_max = r(ii);
    tmp = truncate(A,10^-12,r_op_max,r_op_min);    
    app_tmp = apply_operator_nonglobal(X_bin,tmp,L);
    
    tmp{end} = -tmp{end};
    app_tmp{end} = -app_tmp{end};
    
    E = Add_TTN(A,tmp,tau_bin);
    err(ii) = sqrt(abs(Mat0Mat0(E,E)));
    E2 = Add_TTN(app,app_tmp,tau_bin);
    err_app(ii) = sqrt(abs(Mat0Mat0(E2,E2)));
    
    % TT 
    r_op_max = r(ii);
    tmp = truncate(A_TT,10^-12,r_op_max,r_op_min);    
    app_tmp = apply_operator_nonglobal_TT(X_TT,tmp,L);
    
    tmp{end} = -tmp{end};
    app_tmp{end} = -app_tmp{end};
    
    E = Add_TTN(A_TT,tmp,tau_TT);
    err_TT(ii) = sqrt(abs(Mat0Mat0(E,E)));
    E2 = Add_TTN(app_TT,app_tmp,tau_TT);
    err_app_TT(ii) = sqrt(abs(Mat0Mat0(E2,E2)));
    
end

figure(1)
subplot(1,2,1), semilogy(r,err,'LineWidth',2)
xlabel('Max. rank of truncated operator')
ylabel('Error of truncated operator')
title('Binary tree of minimal height')

subplot(1,2,2), semilogy(r,err_TT,'LineWidth',2)
xlabel('Max. rank of truncated operator')
ylabel('Error of truncated operator')
title('Binary tree of maximal height')


% figure(1)
% subplot(1,2,1), semilogy(r,err,'LineWidth',3)
% xlabel('max. rank of truncated binary operator')
% ylabel('error of truncated binary operator ')
% 
% subplot(1,2,2), semilogy(r,err_app,'LineWidth',3)
% xlabel('max. rank of truncated binary operator')
% ylabel('error of truncated binary operator applied to X')
% 
% figure(2)
% subplot(1,2,1), semilogy(r,err_TT,'LineWidth',3)
% xlabel('max. rank of truncated TT operator')
% ylabel('error of truncated TT operator ')
% 
% subplot(1,2,2), semilogy(r,err_app_TT,'LineWidth',3)
% xlabel('max. rank of truncated TT operator')
% ylabel('error of truncated TT operator applied to X')
