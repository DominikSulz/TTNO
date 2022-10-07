function[obs_n,density,max_r,free_p] = long_range_TT(d,r,V,gamma,Delta,Omega,alpha,dt,T_end,tol,r_max,r_min,r_op_max,r_op_min)

% input:
% d = number of particles, for the moment must be d=2^N
% r = bond dimension (must be >1)
% dt = time step size
% Omega,V,\gamma,Delta are the parameters of the model
% T_end = final time
% tol, r_max, r_min are needed for rank-adaptivity
%
% output:
% obs_n = avarage density in n over time 
% density = density for n_1,...,n_d
% Corr = correlation matrix of n-direction

%%% time-step
Iter=T_end/dt;

% initialisations
sx=[0,1;1,0];  %% Pauli Matrix x
sy=[0,-1i;1i,0]; %% Pauli Matrix y
sz=[1,0;0,-1]; %% Pauli Matrix z
n=[1,0;0,0];  %% Projector onto the excited state Pu=(sz+id)/2;
id=[1,0;0,1];  %% Identity for the single spin
J = [0,0;1,0];

density = zeros(1,d);

% create initial data
[X,tau] = init_spin_all_dim_same_rank_TT(d);

% Observable magnetisation 
% Obs_n = make_operator_observable_diss(X,kron(n,id),d);
flat_state = obs_state_TT(d,r(end));

% create cell array for Hamiltonian
B = linearisation_long_range(d,J,sx,n,V,Delta,Omega,gamma,alpha);

% make operator 
A = make_operator_TT(X,B,tau,4*ones(d,1));
A = rounding(A,tau);
A = truncate(A,10^-8,r_op_max,r_op_min);

% make operator observables 
% Op_n = cell(1,d);
% for ii=1:d
% %    Op_n{ii} =  make_operator_observable_diss_corr(X,kron(n,id),d,ii);
%     B = cell(1,d);
%     B{ii} = kron(n,id);
%     Op_n{ii} = make_operator_TT(X,B,tau,4*ones(d,1));
% end

B = cell(d,d);
for ii=1:d
    B{ii,ii} = kron(n,id);
end
Op = make_operator_TT(X,B,tau,4*ones(d,1));


% time evolution
time = [];
obs_n = [];
max_r = max_rank(X);
free_p = count_free_parameter(X);

X_start = X;
for it=1:Iter
    t0 = (it-1)*dt;
    t1 = it*dt;
    time(it)=t1;
    
    % time integration with rank adaptive integrator
    X_new = TTN_integrator_complex_rank_adapt_nonglobal(tau,X_start,@F_Hamiltonian_diss_TT,t0,t1,A,d,r_min);
    X_new = truncate(X_new,tol,r_max,r_min);
    
    max_r(it+1) = max_rank(X_new);
    free_p(it+1) = count_free_parameter(X_new);
    
    % renormalization
    tmp = Mat0Mat0(flat_state,X_new);
    X_new{end} = X_new{end}/tmp;
    
    % setting for next time step
    X_start = X_new;
    
%     % computes density
%     for jj=1:d
%         tmp = apply_operator_nonglobal_TT(X_new,Op_n{jj},d);
%         density(it,jj) = Mat0Mat0(flat_state,tmp);
%     end
    
%     % compute average density in n direction
%     obs_n(it) = (1/d)*sum(density(it,:));
    tmp = apply_operator_nonglobal_TT(X_new,Op,d);
    obs_n(it) = (1/d)*Mat0Mat0(flat_state,tmp);
    
%     tmp_n = apply_operator_nonglobal(X_new,Obs_n,d); % n-direction
%     obs_n(it) = (1/d)*Mat0Mat0(flat_state,tmp_n);
    it
    
end
Corr = 1;
% % Correlations
% Corr = zeros(d,d);
% 
% for k=1:d
%     app_k = apply_operator_nonglobal(X_new,Op_n{k},d);
%     for h=1:d
%         app_h = apply_operator_nonglobal(X_new,Op_n{h},d);
%         
%         % <n^k n^h>
%         app_k_n = apply_operator_nonglobal(app_h,Op_n{k},d);
%         
%         Corr(k,h) = Mat0Mat0(flat_state,app_k_n) ...
%             - Mat0Mat0(flat_state,app_k)*Mat0Mat0(flat_state,app_h);
%     end
% end

end