function [V_mat] = test_V_matrix(d,J,sx,n,V,Delta,Omega,gamma,alpha)

B = cell(d,d);
id = [1,0;0,1];
V_mat = zeros(d,d);

count = 1;
% long range
for ii=1:d
    for jj=1:d
        if ii==jj
            
        else
            V_mat(ii,jj) = 1/abs(ii-jj)^alpha;
            B{count,ii} = -1i*(V/2) * 1/abs(ii-jj)^alpha * kron(n,id);
            B{count,jj} = kron(n,id);
            count = count + 1;
        end
    end
end

end