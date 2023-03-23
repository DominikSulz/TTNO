function [H,mi] = adjust_H(H)

if H.leafnode == 1
    mi = 1;
    if H.U < 0 
        H.U = -H.U;
        mi = mi*(-1);
    end
else 
    [H.A11,mi] = adjust_H(H.A11);
    if mi==-1
        H.B12 = -H.B12;
    end
    [H.A22,mi] = adjust_H(H.A22);
    if mi==-1
        H.B21 = -H.B21;
    end
end

end

