    for ilayer in range(NLAYER):
        for iwave in range(NWAVE):
            bb = calc_planck(wave_grid[iwave], T_layer[ilayer])
            for ig in range(NG):
                tau_cumulative_w_g[iwave,ig] = tau_total_w_g_l[iwave,ig,ilayer] \
                    + tau_cumulative_w_g[iwave,ig]
                # transmission function
                tr_w_g[iwave,ig] = np.exp(-tau_cumulative_w_g[iwave,ig])
                 # blackbody function
                """# vectorised
                for ig in range(NG):
                    spec_w_g[:,ig] = spec_w_g[:,ig]+(tr_old_w_g[:,ig]-tr_w_g[:,ig])*bb[:]
                """
                spec_w_g[iwave,ig] = spec_w_g[iwave,ig] \
                    + (tr_old_w_g[iwave,ig]-tr_w_g[iwave,ig])*bb

                tr_old_w_g = copy(tr_w_g)
