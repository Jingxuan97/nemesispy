
import numpy as np
def sort2g(RA):
    """
    Sort RA into asceding order
    """
    print('RA=',RA)
    N = len(RA) - 1
    print('N=',N)
    IB = np.arange(N+1)
    print('IB',IB)
    L = int(N/2)+1
    print('L',L)
    IR = N
    print('IR',IR)

    while L != 0:
        print('list',RA)
        # at least two elements
        if L > 1:
            print('L>=1, L=', L)
            L = L-1
            print('L=',L)
            RRA = RA[L]
            print('RRA=',RRA)
            IRB = IB[L]
            print('IRB=',IRB)
        elif L==1:
            # only one element
            print('only one element')
            print('IR=',IR)
            RRA = RA[IR]
            IRB = IB[IR]
            RA[IR] = RA[0]
            IB[IR] = IB[0]
            IR = IR - 1
            print('IR=',IR)
            return RA
            """
            if IR == 0:

                RA[0] = RRA
                IB[0] = IRB
                print('return')
                return RA
            # end if
            """
        # end if
        I = L
        print('I=',I)
        J = L+L
        print('J=',J)

        while J<=IR:
            if J<IR:
                if RA[J]<=RA[J+1]:
                    J = J+1
            if RRA < RA[J]:
                RA[I] = RA[J]
                IB[I] = IB[J]
                I = J
                J = J+J
            else:
                J = IR+1
            # end if
        RA[I] = RRA
        IB[I] = IRB
    return RA