def CIRSrad(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Path):

    """
        FUNCTION NAME : CIRSrad()

        DESCRIPTION : This function computes the spectrum given the calculation type

        INPUTS :

            runname :: Name of the Nemesis run
            Variables :: Python class defining the parameterisations and state vector
            Measurement :: Python class defining the measurements
            Atmosphere :: Python class defining the reference atmosphere
            Spectroscopy :: Python class defining the parameters required for the spectroscopic calculations
            Scatter :: Python class defining the parameters required for scattering calculations
            Stellar :: Python class defining the stellar spectrum
            Surface :: Python class defining the surface
            CIA :: Python class defining the Collision-Induced-Absorption cross-sections
            Layer :: Python class defining the layering scheme to be applied in the calculations
            Path :: Python class defining the calculation type and the path

        OPTIONAL INPUTS: none

        OUTPUTS :

            SPECOUT(Measurement.NWAVE,Path.NPATH) :: Output spectrum (non-convolved) in the units given by IMOD

        CALLING SEQUENCE:

            SPECOUT = CIRSrad(runname,Variables,Measurement,Atmosphere,Spectroscopy,Scatter,Stellar,Surface,CIA,Layer,Path)

        MODIFICATION HISTORY : Juan Alday (25/07/2021)

    """

    import matplotlib as matplotlib
    from scipy import interpolate
    from NemesisPy import k_overlap, k_overlapg, planck
    from copy import copy

    #Initialise some arrays
    ###################################

    #Calculating the vertical opacity of each layer
    ######################################################
    ######################################################
    ######################################################
    ######################################################

    #There will be different kinds of opacities:
    #   Continuum opacity due to aerosols coming from the extinction coefficient
    #   Continuum opacity from different gases like H, NH3 (flags in .fla file)
    #   Collision-Induced Absorption
    #   Scattering opacity derived from the particle distribution and the single scattering albedo.
    #        For multiple scattering, this is passed to scattering routines
    #   Line opacity due to gaseous absorption (K-tables or LBL-tables)

    #Calculating the gaseous line opacity in each layer
    ########################################################################################################

    if Spectroscopy.ILBL==2:  #LBL-table

        TAUGAS = np.zeros([Measurement.NWAVE,Spectroscopy.NG,Layer.NLAY,Spectroscopy.NGAS])  #Vertical opacity of each gas in each layer

        #Calculating the cross sections for each gas in each layer
        k = Spectroscopy.calc_klbl(Layer.NLAY,Layer.PRESS/101325.,Layer.TEMP,WAVECALC=Measurement.WAVE)

        for i in range(Spectroscopy.NGAS):
            IGAS = np.where( (Atmosphere.ID==Spectroscopy.ID[i]) & (Atmosphere.ISO==Spectroscopy.ISO[i]) )
            IGAS = IGAS[0]

            #Calculating vertical column density in each layer
            VLOSDENS = Layer.AMOUNT[:,IGAS].T * 1.0e-4 * 1.0e-20   #cm-2

            #Calculating vertical opacity for each gas in each layer
            TAUGAS[:,0,:,i] = k[:,:,i] * VLOSDENS

        #Combining the gaseous opacity in each layer
        TAUGAS = np.sum(TAUGAS,3) #(NWAVE,NG,NLAY)

        del k

    elif Spectroscopy.ILBL==0:    #K-table

        #Calculating the k-coefficients for each gas in each layer
        k_gas = Spectroscopy.calc_k(Layer.NLAY,Layer.PRESS/101325.,Layer.TEMP,WAVECALC=Measurement.WAVE) # (NWAVE,NG,NLAY,NGAS)

        f_gas = np.zeros([Spectroscopy.NGAS,Layer.NLAY])
        utotl = np.zeros(Layer.NLAY)
        for i in range(Spectroscopy.NGAS):
            IGAS = np.where( (Atmosphere.ID==Spectroscopy.ID[i]) & (Atmosphere.ISO==Spectroscopy.ISO[i]) )
            IGAS = IGAS[0]

            f_gas[i,:] = Layer.PP[:,IGAS].T / Layer.PRESS                     #VMR of each radiatively active gas
            utotl[:] = utotl[:] + Layer.AMOUNT[:,IGAS].T * 1.0e-4 * 1.0e-20   #Vertical column density of the radiatively active gases

        #Combining the k-distributions of the different gases in each layer
        k_layer = k_overlap(Measurement.NWAVE,Spectroscopy.NG,Spectroscopy.DELG,Spectroscopy.NGAS,Layer.NLAY,k_gas,f_gas)  #(NWAVE,NG,NLAY)

        #Calculating the opacity of each layer
        TAUGAS = k_layer * utotl   #(NWAVE,NG,NLAY)

        del k_gas
        del k_layer

    else:
        sys.exit('error in CIRSrad :: ILBL must be either 0 or 2')


    #Combining the different kinds of opacity in each layer
    ########################################################################################################

    TAUTOT = np.zeros(TAUGAS.shape) #(NWAVE,NG,NLAY)
    for ig in range(Spectroscopy.NG):
        TAUTOT[:,ig,:] = TAUGAS[:,ig,:] + TAUCIA[:,:] + TAUDUST[:,:] + TAURAY[:,:]

    #Calculating the line-of-sight opacities
    #################################################################################################################

    TAUTOT_LAYINC = TAUTOT[:,:,Path.LAYINC[:,:]] * Path.SCALE[:,:]  #(NWAVE,NG,NLAYIN,NPATH)


    #Step through the different number of paths and calculate output spectrum
    ############################################################################

    #Output paths may be:
    #	      Imod
    #		0	(Atm) Pure transmission
    #		1	(Atm) Absorption (useful for small transmissions)
    #		2	(Atm) Emission. Planck function evaluated at each
    #				wavenumber. NOT SUPPORTED HERE.
    #		3	(Atm) Emission. Planck function evaluated at bin
    #				center.
    #		8	(Combined Cell,Atm) The product of two
    #				previous output paths.
    #		11	(Atm) Contribution function.
    #		13	(Atm) SCR Sideband
    #		14	(Atm) SCR Wideband
    #		15	(Atm) Multiple scattering (multiple models)
    #		16	(Atm) Single scattering approximation.
    #		21	(Atm) Net flux calculation (thermal)
    #		22	(Atm) Limb scattering calculation
    #		23	(Atm) Limb scattering calculation using precomputed
    #			      internal radiation field.
    #		24	(Atm) Net flux calculation (scattering)
    #		25	(Atm) Upwards flux (internal) calculation (scattering)
    #		26	(Atm) Upwards flux (top) calculation (scattering)
    #		27	(Atm) Downwards flux (bottom) calculation (scattering)
    #		28	(Atm) Single scattering approximation (spherical)

    IMODM = np.unique(Path.IMOD)

    if IMODM==0:

        #Calculating the total opacity over the path
        TAUTOT_PATH = np.sum(TAUTOT_LAYINC,2) #(NWAVE,NG,NPATH)

        #Pure transmission spectrum
        SPECOUT = np.exp(-(TAUTOT_PATH))  #(NWAVE,NG,NPATH)

        xfac = 1.0
        if Measurement.IFORM==4:  #If IFORM=4 we should multiply the transmission by solar flux
            Stellar.calc_solar_flux()
            #Interpolating to the calculation wavelengths
            f = interpolate.interp1d(Stellar.VCONV,Stellar.SOLFLUX)
            solflux = f(Measurement.WAVE)
            xfac = solflux
            for ipath in range(npath):
                SPECOUT[:,:,ipat] = SPECOUT[:,:,ipat] * xfac

    elif IMODM==1:

        #Calculating the total opacity over the path
        TAUTOT_PATH = np.sum(TAUTOT_LAYINC,2) #(NWAVE,NG,NPATH)

        #Absorption spectrum (useful for small transmissions)
        SPECOUT = 1.0 - np.exp(-(TAUTOT_PATH)) #(NWAVE,NG,NPATH)

    elif IMODM==3: #Thermal emission from planet

        SPECOUT = np.zeros([Measurement.NWAVE,Spectroscopy.NG,Path.NPATH])

        #Defining the units of the output spectrum
        xfac = 1.
        if Measurement.IFORM==1:
            xfac=np.pi*4.*np.pi*((Atmosphere.RADIUS)*1.0e2)**2.
            f = interpolate.interp1d(Stellar.VCONV,Stellar.SOLSPEC)
            solpspec = f(Measurement.WAVE)  #Stellar power spectrum (W (cm-1)-1 or W um-1)
            xfac = xfac / solpspec

        #Calculating spectrum
        for ipath in range(Path.NPATH):
            #Calculating atmospheric contribution
            taud = np.zeros([Measurement.NWAVE,Spectroscopy.NG])
            trold = np.ones([Measurement.NWAVE,Spectroscopy.NG])
            specg = np.zeros([Measurement.NWAVE,Spectroscopy.NG])

            #
            for j in range(Path.NLAYIN[ipath]):

                taud[:,:] = taud[:,:] + TAUTOT_LAYINC[:,:,j,ipath]
                tr = np.exp(-taud)

                bb = planck(Measurement.ISPACE,Measurement.WAVE,Path.EMTEMP[j,ipath])
                for ig in range(Spectroscopy.NG):
                    specg[:,ig] = specg[:,ig] + (trold[:,ig]-tr[:,ig])*bb[:] * xfac

                trold = copy(tr)



            #Calculating surface contribution

            p1 = Layer.PRESS[Path.LAYINC[int(Path.NLAYIN[ipath]/2)-1,ipath]]
            p2 = Layer.PRESS[Path.LAYINC[int(Path.NLAYIN[ipath]-1),ipath]]

            if p2>p1:  #If not limb path, we add the surface contribution

                if Surface.TSURF<=0.0:
                    radground = planck(Measurement.ISPACE,Measurement.WAVE,Path.EMTEMP[Path.NLAYIN[ipath]-1,ipath])
                else:
                    bbsurf = planck(Measurement.ISPACE,Measurement.WAVE,Surface.TSURF)

                    f = interpolate.interp1d(Surface.VEM,Surface.EMISSIVITY)
                    emissivity = f(Measurement.WAVE)

                    radground = bbsurf * emissivity

                for ig in range(Spectroscopy.NG):
                    specg[:,ig] = specg[:,ig] + trold[:,ig] * radground[:] * xfac

            SPECOUT[:,:,ipath] = specg[:,:]


    #Now integrate over g-ordinates
    SPECOUT = np.tensordot(SPECOUT, Spectroscopy.DELG, axes=([1],[0])) #NWAVE,NPATH

    return SPECOUT
