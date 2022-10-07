clear all; close all; clc

addpath('C:\Users\Dominik\Documents\MATLAB\Low rank approximations\rank_adaptive_integrator_for_TTN')

% number particles
L = 16;

% parameters of the model
Omega = 0.4;
Delta = -2;
gamma = 1;
alpha = 0.01;
c_alpha=sum((1:1:L).^(-alpha));
V = 2/c_alpha;
r = [4 4 4 4 4]; % bottom layer: between 2 and 4

% bond dimension operator
r_op_max_save = [2 3 4 6 8 10 12 14 16 18 20 22 24 26 28 30];
r_op_min = 2;

% initialisations
sx=[0,1;1,0];  %% Pauli Matrix x
sy=[0,-1i;1i,0]; %% Pauli Matrix y
sz=[1,0;0,-1]; %% Pauli Matrix z
n=[1,0;0,0];  %% Projector onto the excited state Pu=(sz+id)/2;
id=[1,0;0,1];  %% Identity for the single spin
J = [0,0;1,0];

% create initial data
[X_TTN,tau_TTN] = init_diss_all_dim_diff_rank(r,2,L);
[X_TT,tau_TT] = init_spin_all_dim_same_rank_TT(L);

% create cell array for Hamiltonian
B = linearisation_long_range(L,J,sx,n,V,Delta,Omega,gamma,alpha);

% make operator 
A_TTN = make_operator(X_TTN,B,tau_TTN,4*ones(L,1));
A_TTN = rounding(A_TTN,tau_TTN);

A_TT = make_operator_TT(X_TT,B,tau_TT,4*ones(L,1));
A_TT = rounding(A_TT,tau_TT);

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



