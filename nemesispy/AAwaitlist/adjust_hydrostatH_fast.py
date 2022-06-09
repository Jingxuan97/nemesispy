import nummpy as np
def adjust_hydrostatH(P, T, ID, VMR, M_plt, R_plt, H=np.array([0])):
    """
    Adjust the input altitude profile H according to hydrostatic equilibrium
    using the input pressure, temperature and VMR profiles and planet mass
    and radius. The routine calls calc_grav_simple to calculate gravity.

    Parameters
    ----------
    H : ndarray
        Altitude profile to be adjusted
    P : ndarray
        Pressure profile
    T : ndarray
        Temperature profile
    ID : ndarray
        Gas ID.
    VMR : ndarray
        Volume mixing ration profile.
    M_plt : real
        Planetary mass
    R_plt : real
        Planetary radius

    Returns
    -------
    H : ndarray
        Adjusted altitude profile satisfying hydrostatic equlibrium.

    """
    # note number of profile points and number of gases
    NPRO,NVMR = VMR.shape
    if not H.any():
        H = np.linspace(0,1e5,NPRO)
    # initialise array for scale heights
    scale_height = np.zeros(NPRO)

    # First find level closest ot zero altitude
    ialt = (np.abs(H - 0.0)).argmin()
    alt0 = H[ialt]
    if ( (alt0>0.0) & (ialt>0)):
        ialt = ialt -1

    #Calculate the mean molecular weight at each level
    xmolwt = np.zeros(NPRO)
    for ipro in range(NPRO):
        xmolwt[ipro] = calc_mmw(ID,VMR[ipro,:])

    # iterate until hydrostatic equilibrium
    xdepth = 2
    adjusted_H = H
    dummy_H = np.zeros(NPRO)
    while xdepth > 1:

        h = np.zeros(NPRO)
        dummy_H[:] = adjusted_H

        #Calculating the atmospheric model depth
        atdepth = dummy_H[-1] - dummy_H[0]

        #Calculate the gravity at each altitude level
        gravity = np.zeros(NPRO)
        gravity[:] =  calc_grav_simple(h=dummy_H, M_plt=M_plt, R_plt=R_plt)

        #Calculate the scale height
        scale = np.zeros(NPRO)
        scale[:] = K_B*T[:]/(xmolwt[:]*gravity[:])

        if ialt > 0 and ialt < NPRO-1 :
            dummy_H[ialt] = 0.0

        # nupper = NPRO - ialt - 1
        for i in range(ialt+1, NPRO):
            sh = 0.5 * (scale[i-1] + scale[i])
            #self.H[i] = self.H[i-1] - sh * np.log(self.P[i]/self.P[i-1])
            dummy_H[i] = dummy_H[i-1] - sh * np.log(P[i]/P[i-1])

        for i in range(ialt-1,-1,-1):
            sh = 0.5 * (scale[i+1] + scale[i])
            #self.H[i] = self.H[i+1] - sh * np.log(self.P[i]/self.P[i+1])
            dummy_H[i] = dummy_H[i+1] - sh * np.log(P[i]/P[i+1])

        atdepth1 = dummy_H[-1] - dummy_H[0]

        xdepth = 100.*abs((atdepth1-atdepth)/atdepth)
        # print('xdepth',xdepth)
        adjusted_H = dummy_H

    return adjusted_H