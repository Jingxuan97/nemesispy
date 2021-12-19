import numpy as np
import sys
class CIA_0:

    def __init__(self, INORMAL=0, NPAIR=9, NT=25, NWAVE=1501):

        """
        Inputs
        ------
        @param INORMAL: int,
            Flag indicating whether the ortho/para-H2 ratio is in equilibrium (0 for 1:1) or normal (1 for 3:1)
        @param NPAIR: int,
            Number of gaseous pairs listed
            (Default = 9 : H2-H2 (eqm), H2-He (eqm), H2-H2 (normal), H2-He (normal), H2-N2, H2-CH4, N2-N2, CH4-CH4, H2-CH4)
        @param NT: int,
            Number of temperature levels over which the CIA data is defined
        @param NWAVE: int,
            Number of spectral points over which the CIA data is defined

        Attributes
        ----------
        @attribute WAVEN: 1D array
            Wavenumber array (NOTE: ALWAYS IN WAVENUMBER, NOT WAVELENGTH)
        @attribute TEMP: 1D array
            Temperature levels at which the CIA data is defined (K)
        @attribute K_CIA: 1D array
            CIA cross sections for each pair at each wavenumber and temperature level

        Methods
        ----------
        CIA_0.read_cia(runname)
        """

        #Input parameters
        self.INORMAL = INORMAL
        self.NPAIR = NPAIR
        self.NT = NT
        self.NWAVE = NWAVE

        # Input the following profiles using the edit_ methods.
        self.WAVEN = None # np.zeros(NWAVE)
        self.TEMP = None # np.zeros(NT)
        self.K_CIA = None #np.zeros(NPAIR,NT,NWAVE)


    def read_cia(self,runname,raddata='/Users/aldayparejo/Documents/Projects/PlanetaryScience/NemesisPy-dist/NemesisPy/Data/cia/'):
        """
        Read the .cia file
        @param runname: str
            Name of the NEMESIS run
        """

        from scipy.io import FortranFile

        #Reading .cia file
        f = open(runname+'.cia','r')
        s = f.readline().split()
        cianame = s[0]
        s = f.readline().split()
        dnu = float(s[0])
        s = f.readline().split()
        npara = int(s[0])
        f.close()

        if npara!=0:
            sys.exit('error in read_cia :: routines have not been adapted yet for npara!=0')

        #Reading the actual CIA file
        if npara==0:
            NPAIR = 9

        f = FortranFile(raddata+cianame, 'r' )
        TEMPS = f.read_reals( dtype='float64' )
        KCIA_list = f.read_reals( dtype='float32' )
        NT = len(TEMPS)
        NWAVE = int(len(KCIA_list)/NT/NPAIR)

        NU_GRID = np.linspace(0,dnu*(NWAVE-1),NWAVE)
        K_CIA = np.zeros([NPAIR, NT, NWAVE])

        index = 0
        for iwn in range(NWAVE):
            for itemp in range(NT):
                for ipair in range(NPAIR):
                    K_CIA[ipair,itemp,iwn] = KCIA_list[index]
                    index += 1

        self.NWAVE = NWAVE
        self.NT = NT
        self.NPAIR = NPAIR
        self.WAVEN = NU_GRID
        self.TEMP = TEMPS
        self.K_CIA = K_CIA

    def calc_tau_cia(self,ISPACE,WAVEC,Atmosphere,Layer,MakePlot=False):
        """
        Calculate the CIA opacity in each atmospheric layer
        @param ISPACE: int
            Flag indicating whether the calculation must be performed in wavenumbers (0) or wavelength (1)
        @param WAVEC: int
            Wavenumber (cm-1) or wavelength array (um)
        @param Atmosphere: class
            Python class defining the reference atmosphere
        @param Layer: class
            Layer :: Python class defining the layering scheme to be applied in the calculations

        Outputs
        ________

        TAUCIA(NWAVE,NLAY) :: CIA optical depth in each atmospheric layer
        dTAUCIA(NWAVE,NLAY,7) :: Rate of change of CIA optical depth with:
                                 (1) H2 vmr
                                 (2) He vmr
                                 (3) N2 vmr
                                 (4) CH4 vmr
                                 (5) CO2 vmr
                                 (6) Temperature
                                 (7) para-H2 fraction
        IABSORB(5) :: Flag set to gas number in reference atmosphere for the species whose gradient is calculated
        """

        from scipy import interpolate
        from NemesisPy import find_nearest

#       the mixing ratios of the species contributing to CIA
        qh2=np.zeros(Layer.NLAY)
        qhe=np.zeros(Layer.NLAY)
        qn2=np.zeros(Layer.NLAY)
        qch4=np.zeros(Layer.NLAY)
        qco2=np.zeros(Layer.NLAY)
        IABSORB = np.ones(5,dtype='int32') * -1
        for i in range(Atmosphere.NVMR):

            if Atmosphere.ID[i]==39:
                if((Atmosphere.ISO[i]==0) or (Atmosphere.ISO[i]==1)):
                    qh2[:] = Layer.PP[:,i] / Layer.PRESS[:]
                    IABSORB[0] = i

            if Atmosphere.ID[i]==40:
                qhe[:] = Layer.PP[:,i] / Layer.PRESS[:]
                IABSORB[1] = i

            if Atmosphere.ID[i]==22:
                qn2[:] = Layer.PP[:,i] / Layer.PRESS[:]
                IABSORB[2] = i

            if Atmosphere.ID[i]==6:
                if((Atmosphere.ISO[i]==0) or (Atmosphere.ISO[i]==1)):
                    qch4[:] = Layer.PP[:,i] / Layer.PRESS[:]
                    IABSORB[3] = i

            if Atmosphere.ID[i]==2:
                qco2[:] = Layer.PP[:,i] / Layer.PRESS[:]
                IABSORB[4] = i

#       calculating the opacity
        XLEN = Layer.DELH * 1.0e2  #cm
        TOTAM = Layer.TOTAM * 1.0e-4 #cm-2
        AMAGAT = 2.68675E19 #mol cm-3

        amag1 = (Layer.TOTAM*1.0e-4/XLEN)/AMAGAT  #Number density in AMAGAT units
        tau = XLEN*amag1**2

        #Defining the calculation wavenumbers
        if ISPACE==0:
            WAVEN = WAVEC
        elif ISPACE==1:
            WAVEN = 1.e4/WAVEC
            isort = np.argsort(WAVEN)
            WAVEN = WAVEN[isort]

        if((WAVEN.min()<self.WAVEN.min()) or (WAVEN.max()>self.WAVEN.max())):
            print('warning in CIA :: Calculation wavelengths expand a larger range than in .cia file')

#       calculating the CIA opacity at the correct temperature and wavenumber
        NWAVEC = len(WAVEC)   #Number of calculation wavelengths
        tau_cia_layer = np.zeros([NWAVEC,Layer.NLAY])
        dtau_cia_layer = np.zeros([NWAVEC,Layer.NLAY,7])
        for ilay in range(Layer.NLAY):

            #Interpolating to the correct temperature
            temp1 = Layer.TEMP[ilay]
            temp0,it = find_nearest(self.TEMP,temp1)

            if self.TEMP[it]>=temp1:
                ithi = it
                if it==0:
                    temp1 = self.TEMP[it]
                    itl = 0
                    ithi = 1
                else:
                    itl = it - 1

            elif self.TEMP[it]<temp1:
                itl = it
                if it==self.NT-1:
                    temp1 = self.TEMP[it]
                    ithi = self.NT - 1
                    itl = self.NT - 2
                else:
                    ithi = it + 1

            ktlo = self.K_CIA[:,itl,:]
            kthi = self.K_CIA[:,ithi,:]

            fhl = (temp1 - self.TEMP[itl])/(self.TEMP[ithi] - self.TEMP[itl])
            fhh = (self.TEMP[ithi] - temp1)/(self.TEMP[ithi] - self.TEMP[itl])
            dfhldT = 1./(self.TEMP[ithi] - self.TEMP[itl])
            dfhhdT = -1./(self.TEMP[ithi] - self.TEMP[itl])

            kt = ktlo*(1.-fhl) + kthi * (1.-fhh)
            dktdT = -ktlo * dfhldT - kthi * dfhhdT

            #Cheking that interpolation can be performed to the calculation wavenumbers
            inwave = np.where( (self.WAVEN>=WAVEN.min()) & (self.WAVEN<=WAVEN.max()) )
            inwave = inwave[0]
            if len(inwave)>0:

                k_cia = np.zeros([NWAVEC,self.NPAIR])
                dkdT_cia = np.zeros([NWAVEC,self.NPAIR])
                inwave1 = np.where( (WAVEN>=self.WAVEN.min()) & (WAVEN<=self.WAVEN.max()) )
                inwave1 = inwave1[0]

                #fig,(ax1,ax2) = plt.subplots(2,1,figsize=(10,6))
                #labels = ['H2-H2 (eqm)','H2-He (eqm)','H2-H2 (normal)','H2-He (normal)','H2-N2','H2-CH4','N2-N2','CH4-CH4','H2-CH4)']
                for ipair in range(self.NPAIR):
                    #ax1.plot(self.WAVEN,kt[ipair,:],label=labels[ipair])
                    f = interpolate.interp1d(self.WAVEN,kt[ipair,:])
                    k_cia[inwave1,ipair] = f(WAVEN[inwave1])
                    f = interpolate.interp1d(self.WAVEN,dktdT[ipair,:])
                    dkdT_cia[inwave1,ipair] = f(WAVEN[inwave1])
                    #ax2.plot(WAVEN,k_cia[:,ipair])
                #plt.tight_layout()
                #plt.show()

                #Combining the CIA absorption of the different pairs (included in .cia file)
                sum1 = np.zeros(NWAVEC)
                if self.INORMAL==0:   #equilibrium hydrogen
                    sum1[:] = sum1[:] + k_cia[:,0] * qh2[ilay] * qh2[ilay] + k_cia[:,1] * qhe[ilay] * qh2[ilay]
                    dtau_cia_layer[:,ilay,0] = dtau_cia_layer[:,ilay,0] + 2.*qh2[ilay]*k_cia[:,0] + qhe[ilay]*k_cia[:,1]
                    dtau_cia_layer[:,ilay,1] = dtau_cia_layer[:,ilay,1] + qh2[ilay]*k_cia[:,1]
                    dtau_cia_layer[:,ilay,5] = dtau_cia_layer[:,ilay,5] + qh2[ilay] * qh2[ilay] * dkdT_cia[:,0] + dkdT_cia[:,1] * qhe[ilay] * qh2[ilay]

                elif self.INORMAL==1: #'normal' hydrogen
                    sum1[:] = sum1[:] + k_cia[:,2] * qh2[ilay] * qh2[ilay] + k_cia[:,3] * qhe[ilay] * qh2[ilay]
                    dtau_cia_layer[:,ilay,0] = dtau_cia_layer[:,ilay,0] + 2.*qh2[ilay]*k_cia[:,2] + qhe[ilay]*k_cia[:,3]
                    dtau_cia_layer[:,ilay,1] = dtau_cia_layer[:,ilay,1] + qh2[ilay]*k_cia[:,3]
                    dtau_cia_layer[:,ilay,5] = dtau_cia_layer[:,ilay,5] + qh2[ilay] * qh2[ilay] * dkdT_cia[:,2] + dkdT_cia[:,3] * qhe[ilay] * qh2[ilay]

                sum1[:] = sum1[:] + k_cia[:,4] * qh2[ilay] * qn2[ilay]
                dtau_cia_layer[:,ilay,0] = dtau_cia_layer[:,ilay,0] + qn2[ilay] * k_cia[:,4]
                dtau_cia_layer[:,ilay,2] = dtau_cia_layer[:,ilay,2] + qh2[ilay] * k_cia[:,4]
                dtau_cia_layer[:,ilay,5] = dtau_cia_layer[:,ilay,5] + qn2[ilay]*qh2[ilay] * dkdT_cia[:,4]

                sum1[:] = sum1[:] + k_cia[:,5] * qn2[ilay] * qch4[ilay]
                dtau_cia_layer[:,ilay,2] = dtau_cia_layer[:,ilay,2] + qch4[ilay] * k_cia[:,5]
                dtau_cia_layer[:,ilay,3] = dtau_cia_layer[:,ilay,3] + qn2[ilay] * k_cia[:,5]
                dtau_cia_layer[:,ilay,5] = dtau_cia_layer[:,ilay,5] + qn2[ilay]*qch4[ilay] * dkdT_cia[:,5]

                sum1[:] = sum1[:] + k_cia[:,6] * qn2[ilay] * qn2[ilay]
                dtau_cia_layer[:,ilay,2] = dtau_cia_layer[:,ilay,2] + 2.*qn2[ilay] * k_cia[:,6]
                dtau_cia_layer[:,ilay,5] = dtau_cia_layer[:,ilay,5] + qn2[ilay]*qn2[ilay] * dkdT_cia[:,6]

                sum1[:] = sum1[:] + k_cia[:,7] * qch4[ilay] * qch4[ilay]
                dtau_cia_layer[:,ilay,3] = dtau_cia_layer[:,ilay,3] + 2.*qch4[ilay] * k_cia[:,7]
                dtau_cia_layer[:,ilay,5] = dtau_cia_layer[:,ilay,5] + qch4[ilay]*qch4[ilay] * dkdT_cia[:,7]

                sum1[:] = sum1[:] + k_cia[:,8] * qh2[ilay] * qch4[ilay]
                dtau_cia_layer[:,ilay,0] = dtau_cia_layer[:,ilay,0] + qch4[ilay] * k_cia[:,8]
                dtau_cia_layer[:,ilay,3] = dtau_cia_layer[:,ilay,3] + qh2[ilay] * k_cia[:,8]
                dtau_cia_layer[:,ilay,5] = dtau_cia_layer[:,ilay,5] + qch4[ilay]*qh2[ilay] * dkdT_cia[:,8]

                #Look up CO2-CO2 CIA coefficients (external)
                k_co2 = co2cia(WAVEN)
                sum1[:] = sum1[:] + k_co2[:] * qco2[ilay] * qco2[ilay]
                dtau_cia_layer[:,ilay,4] = dtau_cia_layer[:,ilay,4] + 2.*qco2[ilay]*k_co2[:]

                #Look up N2-N2 NIR CIA coefficients


                #Look up N2-H2 NIR CIA coefficients



                tau_cia_layer[:,ilay] = sum1[:] * tau[ilay]
                dtau_cia_layer[:,ilay,:] = dtau_cia_layer[:,ilay,:] * tau[ilay]


        if ISPACE==1:
            tau_cia_layer[:,:] = tau_cia_layer[isort,:]
            dtau_cia_layer[:,:,:] = dtau_cia_layer[isort,:,:]

        if MakePlot==True:

            fig,ax1 = plt.subplots(1,1,figsize=(10,3))
            for ilay in range(Layer.NLAY):
                ax1.plot(WAVEC,tau_cia_layer[:,ilay])
            ax1.grid()
            plt.tight_layout()
            plt.show()

        return tau_cia_layer,dtau_cia_layer,IABSORB