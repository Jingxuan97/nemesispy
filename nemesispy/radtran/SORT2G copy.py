
import numpy as np
def index(N):
    return N-1
def sort2g(RA):
    """
    Modified numerical recipes routine to sort a vector RA of length N
    into ascending order. Integer vector IB is initially set to
    1,2,3,... and on output keeps a record of how RA has been sorted.
    """
    # print('RA=',RA)
    N = len(RA)
    # print('N=',N)
    IB = np.arange(1,N+1)
    # print('IB',IB)
    L = int(N/2)+1
    # print('L',L)
    IR = N
    # print('IR',IR)

    while True:
        # print('list',RA)
        # at least two elements
        if L > 1:
            # print('L>=1, L=', L)
            L = L-1
            # print('L=',L)
            RRA = RA[index(L)]
            # print('RRA=',RRA)
            IRB = IB[index(L)]
            # print('IRB=',IRB)
        else:
            # only one element
            # print('else')
            # print('IR=',IR)
            RRA = RA[index(IR)]
            IRB = IB[index(IR)]
            RA[index(IR)] = RA[index(1)]
            IB[index(IR)] = IB[index(1)]
            IR = IR - 1
            # print('IR=',IR)
            if IR == 1:
                RA[index(1)] = RRA
                IB[index(1)] = IRB
                # print('return')
                return RA,IB-1
            # end if
        # end if
        I = L
        # print('I=',I)
        J = L+L
        # print('J=',J)

        while J<=IR:
            if J<IR:
                if RA[index(J)]<=RA[index(J+1)]:
                    J = J+1
            if RRA < RA[index(J)]:
                RA[index(I)] = RA[index(J)]
                IB[index(I)] = IB[index(J)]
                I = J
                J = J+J
            else:
                J = IR+1
            # end if
        RA[index(I)] = RRA
        IB[index(I)] = IRB
        # print('end of loop')
        # print('RA=', RA)
