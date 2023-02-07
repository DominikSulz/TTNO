function [TTNO] = TTNO_HSS_construct(H,A,l,num_l,n,pos)

TTNO = cell(1,4);
I = eye(n(1),n(1));
I = I(:);

if l==1 % leaf 
    tmp = A{1};
    U = [I tmp(:)];
    TTNO{1} = U;
    
    tmp = A{2};
    U = [I tmp(:)];
    TTNO{2} = U;
    
    if l ~= num_l 
        mat_C = zeros(4,3);
        mat_C(1,1) = 1; % identity
        mat_C(2,2) = H.Rl; % A1
        mat_C(3,2) = H.Rr; % A2
        mat_C(4,3) = H.B12;
        TTNO{3} = eye(3,3);
        TTNO{4} = mat2tens(mat_C.',[2 2 3],1:2,3);  
        TTNO{4} = tensor(TTNO{4},[2 2 3]);
    else
        TTNO{3} = 1;
        C = zeros(2,2);
        C(2,2) = H.B12;
        TTNO{4} = tensor(C,[2 2 1]);
    end
    
elseif (l>1) && (l<num_l) % intermediate tensors
    
else % root tensor
    len = length(A);
    s = len/2;
    TTNO{1} = TTNO_HSS_construct(H.A11,A(1:s),l-1,num_l,n(1:s),pos(1:s));
    TTNO{2} = TTNO_HSS_construct(H.A22,A((s+1):len),l-1,num_l,n((s+1):len),pos((s+1):len));
    
    mat_C = zeros();
    
    
    
    mat_C = mat2tens(mat_C.',[2^(l-1)+2 2^(l-1)+2 1],3,1:2); 
    TTNO{end} = tensor(mat_C,[2^(l-1)+2 2^(l-1)+2 1]);
    
end

end