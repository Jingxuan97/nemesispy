def calc_k(filename,wavemin,wavemax,npoints,press,temp,MakePlot=False):

    """

        FUNCTION NAME : calc_k()

        DESCRIPTION : Calculate the k coefficients of a gas at a given pressure and temperature
                      looking at pre-tabulated correlated-k tables

        INPUTS :

            filename :: Name of the file (supposed to have a .lta extension)
            wavemin :: Wavenumbers to calculate the spectrum (cm-1)
            wavemax :: Maximum Wavenumber to calculate the spectrum (cm-1)
            npoints :: Number of p-T levels at which the absorption coefficient must be computed
            press(npoints) :: Pressure (atm)
            temp(npoints) :: Temperature (K)

        OPTIONAL INPUTS:

            MakePlot :: If True, a summary plot is generated

        OUTPUTS :

            wavek :: Calculation wavenumbers (cm-1)
            ng :: Number of g-ordinates
            g_ord :: G-ordinates
            del_g :: Interval between contiguous g-ordinates
            k(nwave,ng,npoints) :: K coefficients

        CALLING SEQUENCE:

            wavekta,ng,g_ord,del_g,k = calc_k(filename,wavemin,wavemax,npoints,press,temp)

        MODIFICATION HISTORY : Juan Alday (25/09/2019)

    """

    from NemesisPy import find_nearest

    gasID,isoID,nwave,wave,fwhm,ng,g_ord,del_g,npress,presslevels,ntemp,templevels,k_g = read_ktable(filename,wavemin,wavemax)

    #Interpolating to the correct pressure and temperature
    ########################################################

    k_good = np.zeros([nwave,ng,npoints])
    for ipoint in range(npoints):
        press1 = press[ipoint]
        temp1 = temp[ipoint]

        #Getting the levels just above and below the desired points
        lpress  = np.log(press1)
        press0,ip = find_nearest(presslevels,press1)

        if presslevels[ip]>=press1:
            iphi = ip
            if ip==0:
                ipl = 0
            else:
                ipl = ip - 1
        elif presslevels[ip]<press1:
            ipl = ip
            if ip==npress-1:
                iphi = npress - 1
            else:
                iphi = ip + 1

        temp0,it = find_nearest(templevels,temp1)

        if templevels[it]>=temp1:
            ithi = it
            if it==0:
                itl = 0
            else:
                itl = it - 1
        elif templevels[it]<temp1:
            itl = it
            if it==ntemp-1:
                ithi = ntemp - 1
            else:
                ithi = it + 1

        plo = np.log(presslevels[ipl])
        phi = np.log(presslevels[iphi])
        tlo = templevels[itl]
        thi = templevels[ithi]
        klo1 = np.zeros([nwave,ng])
        klo2 = np.zeros([nwave,ng])
        khi1 = np.zeros([nwave,ng])
        khi2 = np.zeros([nwave,ng])
        klo1[:] = k_g[:,:,ipl,itl]
        klo2[:] = k_g[:,:,ipl,ithi]
        khi2[:] = k_g[:,:,iphi,ithi]
        khi1[:] = k_g[:,:,iphi,itl]

        #Interpolating to get the k-coefficients at desired p-T
        if ipl==iphi:
            v = 0.5
        else:
            v = (lpress-plo)/(phi-plo)

        if itl==ithi:
            u = 0.5
        else:
            u = (temp1-tlo)/(thi-tlo)

        k_good[:,:,ipoint] = np.exp((1.0-v)*(1.0-u)*np.log(klo1[:,:]) + v*(1.0-u)*np.log(khi1[:,:]) + v*u*np.log(khi2[:,:]) + (1.0-v)*u*np.log(klo2[:,:]))

    if MakePlot==True:
        fig, ax = plt.subplots(1,1,figsize=(10,6))

        k_abs = np.matmul(k_good[:,:,npoints-1], del_g)
        k_abslo1 = np.matmul(klo1[:,:], del_g)
        k_abslo2 = np.matmul(klo2[:,:], del_g)
        k_abshi1 = np.matmul(khi1[:,:], del_g)
        k_abshi2 = np.matmul(khi2[:,:], del_g)
        ax.semilogy(wave,k_abslo1,label='p = '+str(np.exp(plo))+' atm - T = '+str(tlo)+' K')
        ax.semilogy(wave,k_abslo2,label='p = '+str(np.exp(plo))+' atm - T = '+str(thi)+' K')
        ax.semilogy(wave,k_abshi1,label='p = '+str(np.exp(phi))+' atm - T = '+str(tlo)+' K')
        ax.semilogy(wave,k_abshi2,label='p = '+str(np.exp(phi))+' atm - T = '+str(thi)+' K')
        ax.semilogy(wave,k_abs,label='p = '+str(press1)+' atm - T = '+str(temp1)+' K',color='black')
        ax.legend()
        ax.grid()
        plt.tight_layout()
        plt.show()

    return wave,ng,g_ord,del_g,k_good

def k_overlap_v3(nwave,ng,del_g,ngas,npoints,k_gas,f):

    """

        FUNCTION NAME : k_overlap()

        DESCRIPTION : This subroutine combines the absorption coefficient distributions of
                      several overlapping gases. The overlap is implicitly assumed to be random
                      and the k-distributions are assumed to have NG-1 mean values and NG-1
                      weights. Correspondingly there are NG ordinates in total.

        INPUTS :

            nwave :: Number of wavelengths
            ng :: Number of g-ordinates
            del_g :: Intervals of g-ordinates
            ngas :: Number of gases to combine
            npoints :: Number of p-T points over to run the overlapping routine
            k_gas(nwave,ng,ngas,npoints) :: K-distributions of the different gases
            f(ngas,npoints) :: fraction of the different gases at each of the p-T points


        OPTIONAL INPUTS: None

        OUTPUTS :

            k(nwave,ng,npoints) :: Combined k-distribution

        CALLING SEQUENCE:

            k = k_overlap(nwave,ng,del_g,ngas,npoints,k_gas,f)

        MODIFICATION HISTORY : Juan Alday (25/09/2019)

    """

    from copy import copy

    k = np.zeros([nwave,ng,npoints])

    if ngas<=1:  #There are not enough gases to combine
        k[:,:,:] = k_gas[:,:,0,:]
    else:

        for ip in range(npoints): #running for each p-T case

            for igas in range(ngas-1):

                print(ip,igas)

                #getting first and second gases to combine
                if igas==0:
                    k_gas1 = np.zeros([nwave,ng])
                    k_gas2 = np.zeros([nwave,ng])
                    k_gas1[:,:] = copy(k_gas[:,:,ip,igas])
                    k_gas2[:,:] = copy(k_gas[:,:,ip,igas+1])
                    f1 = f[igas,ip]
                    f2 = f[igas+1,ip]

                    k_temp = np.zeros([nwave,ng])
                else:
                    k_gas1 = copy(k_temp)
                    k_gas2[:,:] = copy(k_gas[:,:,ip,igas+1])
                    f1 = f_temp
                    f2 = f[igas+1,ip]

                    k_temp = np.zeros([nwave,ng])

                #abort if first abundance = 0.0
                if ((f1==0.0) & (f2==0.0)):
                    f_temp = f1 + f2
                    continue

                if ((f1==0.0) & (f2!=0.0)):
                    k_temp[:,:]=k_gas2[:,:]*f2/(f1+f2)
                    f_temp = f1 + f2
                    continue

                if ((f1!=0.0) & (f2==0.0)):
                    k_temp[:,:]=k_gas1[:,:]*f1/(f1+f2)
                    f_temp = f1 + f2
                    continue

                #abort if first k-distribution = 0.0
                iboth = np.where( (k_gas1[:,ng-1]==0.0) & (k_gas2[:,ng-1]==0.0))
                iboth = iboth[0]

                k_temp[iboth,:] = 0.0

                i1 = np.where( (k_gas1[:,ng-1]==0.0) & (k_gas2[:,ng-1]!=0.0))
                i1 = i1[0]

                k_temp[i1,:] = k_gas2[i1,:]*f2/(f1+f2)

                i2 = np.where( (k_gas2[:,ng-1]==0.0) & (k_gas1[:,ng-1]!=0.0))
                i2 = i2[0]

                k_temp[i2,:] = k_gas1[i2,:]*f1/(f1+f2)

                #calculating weights and contributions
                weight = np.zeros(ng*ng)
                contrib = np.zeros([nwave,ng*ng])
                iloop = 0
                for i in range(ng):
                    for j in range(ng):
                        weight[iloop] = del_g[i] * del_g[j]
                        contrib[:,iloop] = (k_gas1[:,i]*f1 + k_gas2[:,j]*f2)/(f1+f2)
                        iloop = iloop + 1

                #getting the cumulative g ordinate
                g_ord = np.zeros(ng+1)
                g_ord[0] = 1
                for ig in range(ng):
                    g_ord[ig+1] = g_ord[ig] + del_g[ig]

                if g_ord[ng]<1.0:
                    g_ord[ng] = 1.0

                isort = np.argsort(contrib,axis=1)

                igood = np.where( (k_gas1[:,ng-1]!=0.0) & (k_gas2[:,ng-1]!=0.0))
                igood = igood[0]
                #for iwave1 in range(len(igood)):
                for iwave1 in range(nwave):

                    #iwave = igood[iwave1]
                    iwave = iwave1

                    #ranking contrib and weight arrays in ascending order of k (i.e. cont values)
                    isort = np.argsort(contrib[iwave,:])
                    contrib1 = contrib[iwave,isort]
                    weight1 = weight[isort]

                    #Now form new g(k) by summing over weight
                    gdist = np.zeros(ng*ng)
                    gdist[0] = 0.0
                    for i in range(ng*ng-1):
                        gdist[i+1] = weight[i+1] + gdist[i]

                    ig = 0
                    sum1 = 0.0
                    kg = np.zeros(ng)
                    for i in range(ng*ng):

                        if ((gdist[i]<g_ord[ig+1]) & (ig<=ng-1)):
                            kg[ig] = kg[ig] + contrib1[i] * weight1[i]
                            sum1 = sum1 + weight1[i]
                        else:
                            frac = (g_ord[ig+1]-gdist[i-1])/(gdist[i]-gdist[i-1])
                            kg[ig] = kg[ig] + frac * contrib1[i] * weight1[i]
                            sum1 = sum1 + frac * weight1[i]
                            kg[ig] = kg[ig] / sum1
                            ig = ig + 1
                            if(ig<=ng-1):
                                sum1 = (1.-frac) * weight1[i]
                                kg[ig] = kg[ig] + (1.-frac) * contrib1[i] * weight1[i]

                    if(ig==ng-1):
                        kg[ig] = kg[ig]/sum1

                    k_temp[iwave,:] = kg[:]
                    f_temp = f1 + f2

            k[:,:,ip] = k_temp[:,:]

    return k

###############################################################################################
def k_overlap(nwave,ng,del_g,ngas,npoints,k_gas,f):

    """

        FUNCTION NAME : k_overlap()

        DESCRIPTION : This subroutine combines the absorption coefficient distributions of
                      several overlapping gases. The overlap is implicitly assumed to be random
                      and the k-distributions are assumed to have NG-1 mean values and NG-1
                      weights. Correspondingly there are NG ordinates in total.

        INPUTS :

            nwave :: Number of wavelengths
            ng :: Number of g-ordinates
            del_g :: Intervals of g-ordinates
            ngas :: Number of gases to combine
            npoints :: Number of p-T points over to run the overlapping routine
            k_gas(nwave,ng,ngas,npoints) :: K-distributions of the different gases
            f(ngas,npoints) :: fraction of the different gases at each of the p-T points


        OPTIONAL INPUTS: None

        OUTPUTS :

            k(nwave,ng,npoints) :: Combined k-distribution

        CALLING SEQUENCE:

            k = k_overlap(nwave,ng,del_g,ngas,npoints,k_gas,f)

        MODIFICATION HISTORY : Juan Alday (25/09/2019)

    """

    k = np.zeros((nwave,ng,npoints))

    if ngas<=1:  #There are not enough gases to combine
        k[:,:,:] = k_gas[:,:,:,0]
    else:

        for ip in range(npoints): #running for each p-T case

            for igas in range(ngas-1):

                #getting first and second gases to combine
                if igas==0:
                    k_gas1 = np.zeros((nwave,ng))
                    k_gas2 = np.zeros((nwave,ng))
                    k_gas1[:,:] = k_gas[:,:,ip,igas]
                    k_gas2[:,:] = k_gas[:,:,ip,igas+1]
                    f1 = f[igas,ip]
                    f2 = f[igas+1,ip]

                    k_combined = np.zeros((nwave,ng))
                else:
                    #k_gas1 = np.zeros((nwave,ng))
                    #k_gas2 = np.zeros((nwave,ng))
                    k_gas1[:,:] = k_combined[:,:]
                    k_gas2[:,:] = k_gas[:,:,ip,igas+1]
                    f1 = f_combined
                    f2 = f[igas+1,ip]

                    k_combined = np.zeros((nwave,ng))

                for iwave in range(nwave):

                    k_g_combined, f_combined = k_overlap_two_gas(k_gas1[iwave,:], k_gas2[iwave,:], f1, f2, del_g)
                    k_combined[iwave,:] = k_g_combined[:]

            k[:,:,ip] = k_combined[:,:]

    return k

def k_overlap_two_gas(k_g1, k_g2, q1, q2, del_g):

    """

        FUNCTION NAME : mix_two_gas_k()

        DESCRIPTION : This subroutine combines the absorption coefficient distributions of
                      two overlapping gases. The overlap is implicitly assumed to be random
                      and the k-distributions are assumed to have NG-1 mean values and NG-1
                      weights. Correspondingly there are NG ordinates in total.

        INPUTS :

            k_g1(ng) :: k-coefficients for gas 1 at a particular wave bin and temperature/pressure.
            k_g2(ng) :: k-coefficients for gas 2 at a particular wave bin and temperature/pressure.
            q1 :: Volume mixing ratio of gas 1
            q2 :: Volume mixing ratio of gas 2
            del_g(ng) ::Gauss quadrature weights for the g-ordinates, assumed same for both gases.


        OPTIONAL INPUTS: None

        OUTPUTS :

            k_g_combine(ng) :: Combined k-distribution of both gases
            q_combined :: Combined Volume mixing ratio of both gases

        CALLING SEQUENCE:

            k_g_combined,VMR_combined = k_overlap_two_gas(k_g1, k_g2, q1, q2, del_g)

        MODIFICATION HISTORY : Juan Alday (25/09/2019)

    """

    ng = len(del_g)  #Number of g-ordinates
    k_g = np.zeros(ng)
    q_combined = q1 + q2

    if((k_g1[ng-1]<=0.0) and (k_g2[ng-1]<=0.0)):
        pass
    elif( (q1<=0.0) and (q2<=0.0) ):
        pass
    elif((k_g1[ng-1]==0.0) or (q1==0.0)):
        k_g[:] = k_g2[:] * q2/(q1+q2)
    elif((k_g2[ng-1]==0.0) or (q2==0.0)):
        k_g[:] = k_g1[:] * q1/(q1+q2)
    else:

        nloop = ng * ng
        weight = np.zeros(nloop)
        contri = np.zeros(nloop)
        ix = 0
        for i in range(ng):
            for j in range(ng):
                weight[ix] = del_g[i] * del_g[j]
                contri[ix] = (k_g1[i]*q1 + k_g2[j]*q2)/(q1+q2)
                ix = ix + 1

        #getting the cumulative g ordinate
        g_ord = np.zeros(ng+1)
        g_ord[0] = 0.0
        for ig in range(ng):
            g_ord[ig+1] = g_ord[ig] + del_g[ig]

        if g_ord[ng]<1.0:
            g_ord[ng] = 1.0

        #sorting contri array
        isort = np.argsort(contri)
        contrib1 = contri[isort]
        weight1 = weight[isort]

        #creating combined g-ordinate array
        gdist = np.zeros(nloop)
        gdist[0] = weight1[0]
        for i in range(nloop-1):
            ix = i + 1
            gdist[ix] = weight1[ix] + gdist[i]

        ig = 0
        sum1 = 0.0
        for i in range(nloop):

            if( (gdist[i]<g_ord[ig+1]) & (ig<=ng-1) ):
                k_g[ig] = k_g[ig] + contrib1[i] * weight1[i]
                sum1 = sum1 + weight1[i]
            else:
                frac = (g_ord[ig+1]-gdist[i-1])/(gdist[i]-gdist[i-1])
                k_g[ig] = k_g[ig] + frac * contrib1[i] * weight1[i]
                sum1 = sum1 + weight1[i]
                k_g[ig] = k_g[ig] / sum1
                ig = ig + 1
                if(ig<=ng-1):
                    sum1 = (1.-frac)*weight1[i]
                    k_g[ig] = k_g[ig] + (1.-frac) * contrib1[i] * weight1[i]

        if ig==ng-1:
            k_g[ig] = k_g[ig] / sum1

    return k_g, q_combined




###############################################################################
@jit(nopython=True)
def k_overlap_multiple_gas(k_gas_g, VMR, g_ord, del_g):
    ngas = k_gas_g.shape[0]
    k_g_combined,VMR_combined = k_gas_g[0,:],VMR[0]
    #mixing in rest of gases inside a loop
    for j in range(1,ngas):
        k_g_combined,VMR_combined\
            = k_overlap_two_gas(k_g_combined,k_gas_g[j,:],VMR_combined,VMR[j],del_g)
    return k_g_combined, VMR_combined

def k_overlap_new(k_gas_w_g_l, del_g, VMR):
    # k_overlap(nwave,ng,del_g,ngas,npoints,k_gas,f)
    """
    This subroutine combines the absorption coefficient distributions of
    several overlapping gases. The overlap is implicitly assumed to be random
    and the k-distributions are assumed to have NG-1 mean values and NG-1
    weights. Correspondingly there are NG ordinates in total.

    Parameters
    ----------
    k_gas_w_g_l : ndarray

        INPUTS :

            nwave :: Number of wavelengths
            ng :: Number of g-ordinates
            del_g :: Intervals of g-ordinates
            ngas :: Number of gases to combine
            npoints :: Number of p-T points over to run the overlapping routine
            k_gas(nwave,ng,ngas,npoints) :: K-distributions of the different gases
            f(ngas,npoints) :: fraction of the different gases at each of the p-T points


        OPTIONAL INPUTS: None

        OUTPUTS :

            k(nwave,ng,npoints) :: Combined k-distribution

        CALLING SEQUENCE:

            k = k_overlap(nwave,ng,del_g,ngas,npoints,k_gas,f)

        MODIFICATION HISTORY : Juan Alday (25/09/2019)

    """
    Ngas, Nwave, Ng, Nlayer = k_gas_w_g_l.shape
    k_wave_g_l = np.zeros((Nwave, Ng, Nlayer))

    if Ngas <= 1: # only one active gas
        k_wave_g_l[:,:,:] = k_gas_w_g_l[0,:,:,:]
    else:
        for ilayer in range(Nlayer): # each atmopsheric layer

            for igas in range(Ngas):



                if igas==0:

                    # k_gas1_w_g = np.zeros((Nwave, Ngas))
                    # k_gas2_w_g = np.zeros((Nwave, Ngas))

                    k_gas1_w_g = k_gas_w_g_l[igas,:,:,ilayer]
                    k_gas2_w_g = k_gas_w_g_l[igas+1,:,:,ilayer]
                    vmr_gas1 = VMR[ilayer,igas]
                    vmr_gas2 = VMR[ilayer,igas+1]
                    k_combined = np.zeros((Nwave,Ng))

                else:
                    k_gas1_w_g = k_combined
                    k_gas2_w_g = k_gas_w_g_l[igas+1,:,:,ilayer]

def k_overlap(k_gas_w_g_l):
    # k_overlap(nwave,ng,del_g,ngas,npoints,k_gas,f)
    """
    This subroutine combines the absorption coefficient distributions of
    several overlapping gases. The overlap is implicitly assumed to be random
    and the k-distributions are assumed to have NG-1 mean values and NG-1
    weights. Correspondingly there are NG ordinates in total.

    Parameters
    ----------
    k_gas_w_g_l : ndarray

        INPUTS :

            nwave :: Number of wavelengths
            ng :: Number of g-ordinates
            del_g :: Intervals of g-ordinates
            ngas :: Number of gases to combine
            npoints :: Number of p-T points over to run the overlapping routine
            k_gas(nwave,ng,ngas,npoints) :: K-distributions of the different gases
            f(ngas,npoints) :: fraction of the different gases at each of the p-T points


        OPTIONAL INPUTS: None

        OUTPUTS :

            k(nwave,ng,npoints) :: Combined k-distribution

        CALLING SEQUENCE:

            k = k_overlap(nwave,ng,del_g,ngas,npoints,k_gas,f)

        MODIFICATION HISTORY : Juan Alday (25/09/2019)

    """
    Ngas, Nwave, Ng, Nlayer =
    k_wave_g_l = np.zeros((Nwave, Ng, Nlayer))
    k = np.zeros((nwave,ng,npoints))

    if ngas<=1:  #There are not enough gases to combine
        k[:,:,:] = k_gas[:,:,:,0]
    else:

        for ip in range(npoints): #running for each p-T case

            for igas in range(ngas-1):

                #getting first and second gases to combine
                if igas==0:
                    k_gas1 = np.zeros((nwave,ng))
                    k_gas2 = np.zeros((nwave,ng))
                    k_gas1[:,:] = k_gas[:,:,ip,igas]
                    k_gas2[:,:] = k_gas[:,:,ip,igas+1]
                    f1 = f[igas,ip]
                    f2 = f[igas+1,ip]

                    k_combined = np.zeros((nwave,ng))
                else:
                    #k_gas1 = np.zeros((nwave,ng))
                    #k_gas2 = np.zeros((nwave,ng))
                    k_gas1[:,:] = k_combined[:,:]
                    k_gas2[:,:] = k_gas[:,:,ip,igas+1]
                    f1 = f_combined
                    f2 = f[igas+1,ip]

                    k_combined = np.zeros((nwave,ng))

                for iwave in range(nwave):

                    k_g_combined, f_combined = k_overlap_two_gas(k_gas1[iwave,:], k_gas2[iwave,:], f1, f2, del_g)
                    k_combined[iwave,:] = k_g_combined[:]

            k[:,:,ip] = k_combined[:,:]

    return k