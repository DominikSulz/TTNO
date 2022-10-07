function [A] = make_operator_TT(X,B,tau,n)
% Input:  X TTN
%         B cell array of dimension s x d - if entry is empty we choose
%         identity
%         n vector containing size of the full tensor
% Output: A operator in TTN format

[s,d]= size(B);

A = set_cores(X,s,d);
A{end} = eye(s,s);
A{end} = tensor(A{end},[s s 1]);
A{end-1} = 1;

for ii=1:d
    U = [];
    for jj=1:s
        if isempty(B{jj,ii}) == 1
            tmp = eye(n(ii),n(ii));
        else
            tmp = B{jj,ii}(:);
        end
        U = [U tmp(:)];
    end
    [Q,S,P] = svd(U);
    rr =rank(S);
    Q = Q(:,1:rr);
    S = S(1:rr,1:rr);
    P = P(:,1:rr);
    R = S*P';

    A = set_operator_TT(A,Q,R,ii,d);
end

end

function [A] = set_operator_TT(Y,Q,R,k,d)
% S is the matrix that shall be on k-th leaf

A = Y;
if d==2 && k<=2 
    A{k} = Q;
    A{end} = ttm(A{end},R,k);
elseif k==1 && d>=3
    A{1} = Q;
    A{end} = ttm(A{end},R,k);
else
    A{2} = set_operator_TT(A{2},Q,R,k-1,d-1);
end

end

function [Y] = set_cores(X,s,d)

if d==1
    Y = X;
    C = eye(s,s);
    C(1) = 1;
    Y{end} = tensor(C,[s s]);
    Y{end-1} = eye(s,s);
else
    Y = X;
    Y{end} = id_tensor([s s s]);
    Y{end-1} = eye(s,s);
end

if 1 == iscell(X)
    m = length(Y) - 2;
    for ii=1:m
        if iscell(Y{ii}) == 1
            Y{ii} = set_cores(Y{ii},s,d-1);
        else
            
        end
    end
end

end

function [C] = id_tensor(s)

% C = zeros(s);
C = sptensor(s);
l = length(s);
for ii=1:s(1)
    if l == 2
        C(ii,ii) = 1;
    elseif l == 3
        C(ii,ii,ii) = 1;
    elseif l == 4
        C(ii,ii,ii,ii) = 1;
    elseif l == 5
        C(ii,ii,ii,ii,ii) = 1;
    elseif l == 6
        C(ii,ii,ii,ii,ii,ii) = 1;    
    end
end
C = tensor(C);
end