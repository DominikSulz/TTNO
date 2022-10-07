function [B] = apply_operator_nonglobal_TT(X,A,d)
% X is a TTN and A a operator in TTN form. B = A(X)

B = X;
for ii=1:d
    B = apply2leaf(B,A,ii,d);
end
B = apply2core(B,A);

end

function [Y] = apply2leaf(Y,A,k,d)
% This function applies the operator A to the k-th leaf of Y

if k==1 && d>=3
    U = A{1};
    dum = [];
    s = size(U);
    for ii=1:s(2)
        M = reshape(U(:,ii),sqrt(s(1)),sqrt(s(1)));
        dum = [dum M*Y{1}];
    end
    Y{1} = dum;
elseif d==2
    U = A{k};
    dum = [];
    s = size(U);
    for ii=1:s(2)
        M = reshape(U(:,ii),sqrt(s(1)),sqrt(s(1)));
        dum = [dum M*Y{k}];
    end
    Y{k} = dum;
else
    Y{2} = apply2leaf(Y{2},A{2},k-1,d-1);
end

% old
% if k==1
%     U = A{1};
%     dum = [];
%     s = size(U);
%     for ii=1:s(2)
%         M = reshape(U(:,ii),sqrt(s(1)),sqrt(s(1)));
%         dum = [dum M*Y{1}];
%     end
%     Y{1} = dum;
% elseif d==1
%     U = A{1};
%     dum = [];
%     s = size(U);
%     for ii=1:s(2)
%         M = reshape(U(:,ii),sqrt(s(1)),sqrt(s(1)));
%         dum = [dum M*Y{k}];
%     end
%     Y{1} = dum;
% else
%     Y{2} = apply2leaf(Y{2},A{2},k-1,d-1);
% end

end


function [Y] = apply2core(Y,A)
% This function applies the operator A to the TTN Y recursevly from the
% root to the leafs
s = size(Y{end});
Y{end} = mult_core(double(A{end}),double(Y{end}));
s2 = size(Y{end});
if s(end) == 1
    Y{end} = tensor(Y{end},[s2 1]);
    Y{end - 1} =  1;
else
    Y{end} = tensor(Y{end});
    Y{end - 1} =  eye(s2(end),s2(end));
end

m = length(Y) - 2;
for ii=1:m
    if 1==iscell(Y{ii})
        Y{ii} = apply2core(Y{ii},A{ii});
    end
end

end


function [value] = mult_core(C1,C2)
% "Multiplies" the cores C1 and C2
tmp1 = size(C1);
tmp2 = size(C2);
d = length(tmp1);

switch d
    
    case 2
        C = kron(C1,C2);
        
    case 3
        C = zeros(tmp1.*tmp2);
        for l=1:tmp1(end)
            for m=1:tmp2(end)
                C(:,:,m+(l-1)*tmp2(end)) = ...
                    kron(C1(:,:,l), C2(:,:,m));
            end
        end
        
    case 4
        C = zeros(tmp1.*tmp2);
        for l=1:tmp1(end)
            for m=1:tmp2(end)
                for p=1:tmp1(end-1)
                    for q=1:tmp2(end-1)
                        C(:,:,q+(p-1)*tmp2(end-1),m+(l-1)*tmp2(end)) = ...
                            kron(C1(:,:,p,l), C2(:,:,q,m));
                    end
                end
            end
        end
end
value = C;
end