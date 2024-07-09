import os
import numpy as np
from numba import njit,prange

MAXMU=21
DNUS0 = 354.69
DNUS1 = 587.07
DNUQ = 4161.0

jcoef = np.array([
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 0],
    [-1, 3, 0, 0, 0],
    [-3, 5, 0, 0, 0],
    [3, -30, 35, 0, 0],
    [15, -70, 63, 0, 0],
    [-5, 105, -315, 231, 0],
    [-35, 315, -693, 429, 0],
    [35, -1260, 6930, -12012, 6435],
    [315, -4620, 18018, -25740, 12155]
], dtype=int)
jdiv = np.array([1, 2, 8, 16, 128], dtype=int)

@njit(fastmath=True, error_model='numpy')
def miescat(xlam, iscat, dsize, nmode, rs, refindx, theta, ntheta, nphas):
    pi = np.pi
    maxth = 100
    maxmode = 10
    pi2 = 1.0 / np.sqrt(2.0 * pi)
    func = np.zeros((4, 2 * maxth))
    phas0 = np.zeros(2 * maxth)
    phas = np.zeros(nphas)
    idump = 0
    numflag = False
    
    if ntheta > maxth:
        print('  TOO MANY ANGLE POINTS: NTHETA =', ntheta, ' MAXTH =', maxth)
    if nmode > maxmode:
        print('  TOO MANY PARTICLE MODES; NMODE =', nmode, ' MAXMODE =', maxmode)
    thetd = np.zeros_like(theta)
    for j in range(ntheta):
        if theta[j] < 0 or theta[j] > 90:
            print(' ANGLE <0 OR > 90')
            return
        thetd[j] = theta[j]
    r1 = rs[0]
    delr = rs[2]
    if rs[1] < rs[0]:
        inr = 1000000001
        cont0 = False
    else:
        inr = 1 + int((rs[1] - rs[0]) / rs[2])
        if inr > 1 and inr % 2 != 0:
            inr += 1
        cont0 = True
    nqmax = np.zeros(nmode)
    if not cont0:
        rmax = np.zeros(nmode)
        for m in range(nmode):
            nqmax[m] = 0.0
            if dsize[m][1] != 0:
                aa = dsize[m][0]
                bb = dsize[m][1]
                alpha = 0.0
                cc = 0.0
                if iscat[0] == 0: # continuous
                    rmax[0] = rs[1]

                if iscat[m] == 1:
                    alpha = dsize[m][2]
                    rmax[m] = alpha * aa * bb
                elif iscat[m] == 2:
                    rmax[m] = np.exp(np.log(aa) - bb ** 2)
                else:
                    cc = dsize[m][2]
                    rmax[m] = (aa / (bb * cc)) ** (1.0 / cc)
    if iscat[0] == 0: # continuous
        r_grid = np.exp(np.linspace(np.log(r1),np.log(rs[1]),dsize.shape[1]))
        r_dist_grid = dsize[0]
        r_dist = np.interp(np.arange(r1,rs[1]+delr,delr),r_grid,r_dist_grid)
    kscat = 0.0
    area = 0.0
    volume = 0.0
    kext = 0.0
    anorm = 0.0
    rfr = refindx[0]
    rfi = refindx[1]
    for m in range(inr):
#     while True:
        rr = r1 + m * delr
        csratio = -1.0
        if csratio == -1.0:
            xx = 2.0 * pi * rr / xlam
            qext, qscat, cq, eltrmx = dmie(xx, rfr, rfi, thetd, ntheta)
        if qext < 0.0:
            numflag = True
        for j in range(1, ntheta + 1):
            for i in range(4):
                func[i, j - 1] = eltrmx[i, j - 1, 0] if qext >= 0.0 else -999.9
        for j in range(ntheta + 1, nphas + 1):
            for i in range(4):
                func[i, j - 1] = eltrmx[i, nphas - j, 1] if qext >= 0.0 else -999.9
        anr = 0.0
        cont = cont0
        for mo in range(1,nmode+1):
            if dsize[mo - 1, 1] != 0:
                aa = dsize[mo - 1, 0]
                bb = dsize[mo - 1, 1]
                alpha = 0.0
                cc = 0.0
                if iscat[mo - 1] == 0:
                    anr1 = r_dist[m]
                elif iscat[mo - 1] == 1:
                    alpha = dsize[mo - 1, 2]
                    anr1 = rr ** alpha * np.exp(-rr / (aa * bb))
                elif iscat[mo - 1] == 3:
                    cc = dsize[mo - 1, 2]
                    anr1 = rr ** aa * np.exp(-bb * rr ** cc)
                else:
                    anr1 = np.sqrt(2 * pi) / (rr * bb) * np.exp(-0.5 * ((np.log(rr) - np.log(aa)) / bb) ** 2)
                anr += anr1
                nqmax[mo - 1] = max(nqmax[mo - 1], anr1 * qscat)
                if not cont:
                    if rr < rmax[mo - 1] or anr1 * qscat > 1e-06 * nqmax[mo - 1]:
                        cont = True

        if m % 2 == 0:
            vv = 2.0 * delr / 3.0
        else:
            vv = 4.0 * delr / 3.0
        if m == 0 or m == inr - 1:
            vv = delr / 3.0
        if qext >= 0:
            for j in range(1, nphas + 1):
                phas0[j - 1] += 0.5 * anr * vv * (func[0, j - 1] + func[1, j - 1])
            kscat += pi * rr * rr * qscat * anr * vv
            kext += pi * rr * rr * qext * anr * vv
            anorm += anr * vv
            area += pi * rr * rr * anr * vv
            volume += 4.0 * pi * rr * rr * rr * anr * vv / 3.0
            if idump == 1:
                print(rr, anr, pi * rr * rr, 1.3333 * np.pi * rr ** 3)
        if anorm > 0.0:
            xscat = float(kscat / anorm * 1e-08)
            xext = float(kext / anorm * 1e-08)
            area *= 1e-08 / anorm
            volume *= 1e-12 / anorm
        else:
            xscat = 0.0
            xext = 0.0
            kscat = 1.0
        for j in range(1, nphas + 1):
            phas[j - 1] = xlam * xlam * float(phas0[j - 1] / (np.pi * kscat))
        if idump == 1:
            print('Volume (cm3) = ', volume)
            print('area (cm2) = ', area)
        if not cont0 and (not cont):
#             print('error?')
            return (xscat, xext, phas)
        m+=1
    return (xscat, xext, phas)

@njit(fastmath=True, error_model='numpy')
def dmie(x, rfr, rfi, thetd, jx):
    ncap = 30000
    acap = np.zeros(ncap, dtype=np.complex128)
    eltrmx = np.zeros((4, 100, 2))
    pi = np.zeros((3, 100))
    tau = np.zeros((3, 100))
    cstht = np.zeros(100)
    si2tht = np.zeros(100)
    t = np.zeros(5)
    taa = np.zeros(2)
    tab = np.zeros(2)
    tb = np.zeros(2)
    tc = np.zeros(2)
    td = np.zeros(2)
    te = np.zeros(2)
    rf = complex(rfr, -rfi)
    rrf = 1.0 / rf
    rx = 1.0 / x
    rrfx = rrf * rx
    t[0] = x * x * (rfr * rfr + rfi * rfi)
    t[0] = np.sqrt(t[0])
    nmx1 = int(1.1 * t[0])
    if not nmx1 < ncap - 1:
        print('LIMIT FOR ACAP IS NOT ENOUGH')
        qext = -1
        return
    nmx2 = int(t[0])
    if not nmx1 > 150:
        nmx1 = 150
        nmx2 = 135
    acap[nmx1] = complex(0)
    for n in range(1, nmx1 + 1):
        nn = nmx1 - n + 1
        acap[nn] = (nn + 1) * rrfx - 1.0 / ((nn + 1) * rrfx + acap[nn + 1])
    for i in range(1, len(acap)):
        acap[i - 1] = acap[i]
    for j in range(jx):
        if thetd[j] < 0.0:
            thetd[j] = abs(thetd[j])
        if thetd[j] == 0.0:
            cstht[j] = 1.0
            si2tht[j] = 0.0
        if thetd[j] > 0.0:
            if thetd[j] < 90.0:
                t[0] = np.pi * thetd[j] / 180.0
                cstht[j] = np.cos(t[0])
                si2tht[j] = 1.0 - cstht[j] * cstht[j]
            elif thetd[j] == 90.0:
                cstht[j] = 0.0
                si2tht[j] = 1.0
            else:
                print('THE VALUE OF THE SCATTERING ANGLE IS GREATER THAN 90.0')
                return
    for j in range(jx):
        pi[0, j] = 0.0
        pi[1, j] = 1.0
        tau[0, j] = 0.0
        tau[1, j] = cstht[j]
    t[0] = np.cos(x)
    t[1] = np.sin(x)
    wm1 = complex(t[0], -t[1])
    wfn1 = complex(t[1], t[0])
    wfn2 = rx * wfn1 - wm1
    tc1 = acap[0] * rrf + rx
    tc2 = acap[0] * rf + rx
    taa[0], taa[1] = (wfn1.real, wfn1.imag)
    tab[0], tab[1] = (wfn2.real, wfn2.imag)
    fna = (tc1 * tab[0] - taa[0]) / (tc1 * wfn2 - wfn1)
    fnb = (tc2 * tab[0] - taa[0]) / (tc2 * wfn2 - wfn1)
    fnap = fna
    fnbp = fnb
    t[0] = 1.5
    tb[0], tb[1] = (fna.real, fna.imag)
    tc[0], tc[1] = (fnb.real, fnb.imag)
    tb *= t[0]
    tc *= t[0]
    for j in range(jx):
        eltrmx[0, j, 0] = tb[0] * pi[1, j] + tc[0] * tau[1, j]
        eltrmx[1, j, 0] = tb[1] * pi[1, j] + tc[1] * tau[1, j]
        eltrmx[2, j, 0] = tc[0] * pi[1, j] + tb[0] * tau[1, j]
        eltrmx[3, j, 0] = tc[1] * pi[1, j] + tb[1] * tau[1, j]
        eltrmx[0, j, 1] = tb[0] * pi[1, j] - tc[0] * tau[1, j]
        eltrmx[1, j, 1] = tb[1] * pi[1, j] - tc[1] * tau[1, j]
        eltrmx[2, j, 1] = tc[0] * pi[1, j] - tb[0] * tau[1, j]
        eltrmx[3, j, 1] = tc[1] * pi[1, j] - tb[1] * tau[1, j]
    qext = 2.0 * (tb[0] + tc[0])
    qscat = (tb[0] ** 2 + tb[1] ** 2 + tc[0] ** 2 + tc[1] ** 2) / 0.75
    ctbrqs = 0.0
    n = 2
    while True:
        t[0] = 2 * n - 1
        t[1] = n - 1
        t[2] = 2 * n + 1
        for j in range(jx):
            pi[2, j] = (t[0] * pi[1, j] * cstht[j] - n * pi[0, j]) / t[1]
            tau[2, j] = cstht[j] * (pi[2, j] - pi[0, j]) - t[0] * si2tht[j] * pi[1, j] + tau[0, j]
        wm1 = wfn1
        wfn1 = wfn2
        wfn2 = t[0] * rx * wfn1 - wm1
        taa[0], taa[1] = (wfn1.real, wfn1.imag)
        tab[0], tab[1] = (wfn2.real, wfn2.imag)
        tc1 = acap[n - 1] * rrf + n * rx
        tc2 = acap[n - 1] * rf + n * rx
        fna = (tc1 * tab[0] - taa[0]) / (tc1 * wfn2 - wfn1)
        fnb = (tc2 * tab[0] - taa[0]) / (tc2 * wfn2 - wfn1)
        t[4] = n
        t[3] = t[0] / (t[4] * t[1])
        t[1] = t[1] * (t[4] + 1.0) / t[4]
        tb[0], tb[1] = (fna.real, fna.imag)
        tc[0], tc[1] = (fnb.real, fnb.imag)
        td[0], td[1] = (fnap.real, fnap.imag)
        te[0], te[1] = (fnbp.real, fnbp.imag)
        ctbrqs += t[1] * (td[0] * tb[0] + td[1] * tb[1] + te[0] * tc[0] + te[1] * tc[1]) + t[3] * (td[0] * te[0] + td[1] * te[1])
        qext += t[2] * (tb[0] + tc[0])
        t[3] = tb[0] ** 2 + tc[0] ** 2 + tb[1] ** 2 + tc[1] ** 2
        qscat += t[2] * t[3]
        t[1] = n * (n + 1)
        t[0] = t[2] / t[1]
        k = n // 2 * 2
        for j in range(jx):
            eltrmx[0, j, 0] += t[0] * (tb[0] * pi[2, j] + tc[0] * tau[2, j])
            eltrmx[1, j, 0] += t[0] * (tb[1] * pi[2, j] + tc[1] * tau[2, j])
            eltrmx[2, j, 0] += t[0] * (tc[0] * pi[2, j] + tb[0] * tau[2, j])
            eltrmx[3, j, 0] += t[0] * (tc[1] * pi[2, j] + tb[1] * tau[2, j])
            if k == n:
                eltrmx[0, j, 1] += t[0] * (-tb[0] * pi[2, j] + tc[0] * tau[2, j])
                eltrmx[1, j, 1] += t[0] * (-tb[1] * pi[2, j] + tc[1] * tau[2, j])
                eltrmx[2, j, 1] += t[0] * (-tc[0] * pi[2, j] + tb[0] * tau[2, j])
                eltrmx[3, j, 1] += t[0] * (-tc[1] * pi[2, j] + tb[1] * tau[2, j])
            else:
                eltrmx[0, j, 1] += t[0] * (tb[0] * pi[2, j] - tc[0] * tau[2, j])
                eltrmx[1, j, 1] += t[0] * (tb[1] * pi[2, j] - tc[1] * tau[2, j])
                eltrmx[2, j, 1] += t[0] * (tc[0] * pi[2, j] - tb[0] * tau[2, j])
                eltrmx[3, j, 1] += t[0] * (tc[1] * pi[2, j] - tb[1] * tau[2, j])
        if t[3] < 1e-14:
            break
        n += 1
        for j in range(jx):
            pi[0, j] = pi[1, j]
            pi[1, j] = pi[2, j]
            tau[0, j] = tau[1, j]
            tau[1, j] = tau[2, j]
        fnap = fna
        fnbp = fnb
        if n <= nmx2:
            continue
        else:
            qext = -1
            print('test')
            return
    for j in range(jx):
        for k in range(2):
            t[:4] = eltrmx[:, j, k]
            eltrmx[0, j, k] = t[2] ** 2 + t[3] ** 2
            eltrmx[1, j, k] = t[0] ** 2 + t[1] ** 2
            eltrmx[2, j, k] = t[0] * t[2] + t[1] * t[3]
            eltrmx[3, j, k] = t[1] * t[2] - t[3] * t[0]
    t[0] = 2.0 * rx * rx
    qext *= t[0]
    qscat *= t[0]
    ctbrqs = 2.0 * ctbrqs * t[0]
    return (qext, qscat, ctbrqs, eltrmx)

@njit(fastmath=True)
def henyey(alpha, f, g1, g2):
    x1 = (1.0 - g1 * g1) / ((1.0 + g1 * g1 - 2 * g1 * alpha) ** 1.5)
    x2 = (1.0 - g2 * g2) / ((1.0 + g2 * g2 - 2 * g2 * alpha) ** 1.5)
    y = f * x1 + (1.0 - f) * x2
    return y

@njit(fastmath=True, error_model='numpy')
def subhgphas(nphase, theta, x):
    max_thet = 100
    pi = np.pi
    cphase = np.zeros(nphase)
    kk = np.zeros((nphase, 3))
    alpha = np.zeros(nphase)
    tphase = np.zeros(nphase)
    xt = np.zeros(3)
    f = x[0]
    g1 = x[1]
    g2 = x[2]
    alpha = np.cos(theta * pi / 180.0)
    cphase = henyey(alpha, f, g1, g2)
    xt[:] = x[:]
    for j in range(3):
        dx = 0.01
        xt[j] = x[j] + dx
        if j == 0:
            if xt[j] > 0.99:
                xt[j] = x[j] - dx
        elif j == 1:
            if xt[j] > 0.98:
                xt[j] = x[j] - dx
        dx = xt[j] - x[j]
        f = xt[0]
        g1 = xt[1]
        g2 = xt[2]
        for i in range(nphase):
            tphase[i] = henyey(alpha[i], f, g1, g2)
        for i in range(nphase):
            kk[i, j] = (tphase[i] - cphase[i]) / dx
        xt[j] = x[j]
    return (cphase, kk)

@njit(fastmath=True)
def mrqcofl(nphase, theta, phase, x):
    mx = 3
    MY = 100
    alpha = np.zeros((mx, mx))
    beta = np.zeros(mx)
    chisq = 0.0
    cphase, kk = subhgphas(nphase, theta, x)
    kk = kk / cphase[:, np.newaxis]
    cphase = np.log(cphase)
    for i in range(nphase):
        dy = phase[i] - cphase[i]
        for j in range(mx):
            wt = kk[i, j]
            for k in range(j + 1):
                alpha[j, k] += wt * kk[i, k]
            beta[j] += dy * wt
        chisq += dy * dy
    for j in range(1, mx):
        for k in range(j):
            alpha[k, j] = alpha[j, k]
    return (alpha, beta, chisq)

@njit(fastmath=True)
def mrqminl(nphase, theta, phase, x, alamda, alpha, beta, chisq, ochisq):
    max_thet = 100
    mx = 3
    my = 100
    covar = np.zeros((mx, mx))
    da = np.copy(beta)[:, None]
    for j in range(mx):
        for k in range(mx):
            covar[j, k] = alpha[j, k]
        covar[j, j] = alpha[j, j] * (1.0 + alamda)
    covar = np.ascontiguousarray(np.linalg.inv(covar))
    da = np.dot(covar, np.ascontiguousarray(da))
    if alamda == 0.0:
        return
    xt = x + da[:, 0]
    for i in range(3):
        if i == 0:
            xt[i] = min(max(xt[i], 1e-06), 0.999999)
        elif i == 1:
            xt[i] = min(max(xt[i], 0.0), 0.98)
        elif i == 2:
            xt[i] = min(max(xt[i], -0.98), -0.1)
    covar, da, chisq = mrqcofl(nphase, theta, phase, xt)
    if chisq <= ochisq:
        alamda *= 0.9
        ochisq = chisq
        alpha[:, :] = covar
        beta[:] = da
        x[:] = xt
    else:
        alamda *= 1.5
        chisq = ochisq
        if alamda > 1e+36:
            alamda = 1e+36
    return (alamda, chisq)

@njit(fastmath=True)
def subfithgm(nphase, theta, phase):
    max_thet = 100
    mx = 3
    my = 100
    x = np.array([0.5, 0.5, -0.5])
    alamda = -1
    nover = 1000
    nc = 0
    ochisq = 0.0
    lphase = np.log(phase)
    for itemp in range(1, nover + 1):
        if alamda < 0:
            alpha, beta, chisq = mrqcofl(nphase, theta, phase, x)
            ochisq = chisq
            alamda = 1000.0
        alamda, chisq = mrqminl(nphase, theta, lphase, x, alamda, alpha, beta, chisq, ochisq)
        if chisq == ochisq:
            nc += 1
            break
        else:
            ochisq = chisq
            nc = 0
    f = x[0]
    g1 = x[1]
    g2 = x[2]
    rms = np.sqrt(chisq)
    return (f, g1, g2, rms)

@njit(fastmath=True)
def get_theta(max_theta):
    ntheta = 1
    theta = np.zeros(max_theta)
    theta[0] = 0.0
    dtheta = 1.0
    while True:
        if theta[ntheta - 1] + dtheta <= 90.0:
            theta[ntheta] = theta[ntheta - 1] + dtheta
            if theta[ntheta] >= 5.0:
                dtheta = 2.5
            if theta[ntheta] >= 20.0:
                dtheta = 5.0
            if theta[ntheta] >= 40.0:
                dtheta = 10.0
            ntheta += 1
        else:
            break
    nphase = 2 * ntheta - 1
    for i in range(ntheta, nphase):
        theta[i] = 180.0 - theta[nphase - i - 1]
    return (theta[:nphase], ntheta, nphase)

@njit(fastmath=True)
def kk_new_sub(vi, k, vm, nm):
    npoints = len(vi)
    va = np.zeros(npoints)
    ka = np.zeros(npoints)
    na = np.zeros(npoints)
    # Reverse order logic
    irev = False
    if vi[0] > vi[-1]:
        va = vi[::-1]
        ka = k[::-1]
        irev = True
    else:
        va = vi
        ka = k

    # Linear interpolation function (verint) to find km at vm
    km = np.interp(vm, va, ka)

    # Integration loop
    for i in range(npoints):
        v = va[i]
        y = np.zeros(npoints)
        for j in range(npoints):
            alpha = va[j]**2 - v**2
            beta = va[j]**2 - vm**2
            if alpha != 0 and beta != 0:
                d1 = ka[j]*va[j] - ka[i]*va[i]
                d2 = ka[j]*va[j] - km*vm
                y[j] = d1/alpha - d2/beta

        # Summation
        sum_ = 0.0
        for l in range(npoints - 1):
            dv = va[l + 1] - va[l]
            sum_ += 0.5 * (y[l] + y[l + 1]) * dv
        na[i] = nm - (2. / np.pi) * sum_

    # Prepare output based on reverse logic
    n = na[::-1] if irev else na

    return n


@njit(parallel=False, cache = os.environ.get("USE_NUMBA_CACHE")=='True', error_model='numpy')
def makephase(wave_grid, iscat, dsize, rs, nimag_wave_grid, calc_wave_grid,
              nreal_ref, nimag, refwave, normwave,downscaling, iwave=1):
    nwave = len(wave_grid)
    data_arr = np.zeros((nwave, 6))
    max_theta = 100
    theta, ntheta, nphase = get_theta(max_theta)
    nreal = kk_new_sub(nimag_wave_grid,nimag,refwave[0],nreal_ref[0])
    refindx_real = np.interp(wave_grid, nimag_wave_grid, nreal)
    refindx_imag = np.interp(wave_grid, nimag_wave_grid, nimag)

    refindx_interp = np.zeros((len(wave_grid),2))
    refindx_interp[:,0] = refindx_real
    refindx_interp[:,1] = refindx_imag
    if downscaling > 0:
        for j in prange(0,nwave,downscaling):
            w = wave_grid[j]
            if iwave == 1:
                xlam = w
            else:
                xlam = 10000.0 / w
            xscat, xext, phase = miescat(xlam, iscat, dsize[None,:], 1, rs, refindx_interp[j], theta[:ntheta], ntheta, nphase)
            omega = xscat / xext
            f, g1, g2, rms = subfithgm(nphase, theta, phase)
            data_arr[j, 0] = xext
            data_arr[j, 1] = xscat
            data_arr[j, 2] = f
            data_arr[j, 3] = g1
            data_arr[j, 4] = g2
            data_arr[j, 5] = rms
            
    elif downscaling == 0:
        for j in prange(len(calc_wave_grid)):
            w = calc_wave_grid[j]
            if iwave == 1:
                xlam = w
            else:
                xlam = 10000.0 / w
            xscat, xext, phase = miescat(xlam, iscat, dsize[None,:], 1, rs, 
                                         np.array([nreal[j%len(nimag_wave_grid)],
                                                   nimag[j%len(nimag_wave_grid)]]), theta[:ntheta], ntheta, nphase)
            omega = xscat / xext
            f, g1, g2, rms = subfithgm(nphase, theta, phase)
            data_arr[j, 0] = xext
            data_arr[j, 1] = xscat
            data_arr[j, 2] = f
            data_arr[j, 3] = g1
            data_arr[j, 4] = g2
            data_arr[j, 5] = rms

#     # Calculating normalising xext - should integrate this with the loop
#     refindx_real = np.interp(normwave[0], nimag_wave_grid, nreal)
#     refindx_imag = np.interp(normwave[0], nimag_wave_grid, nimag)
#     refindx_interp = np.zeros((1,2))
#     refindx_interp[:,0] = refindx_real
#     refindx_interp[:,1] = refindx_imag
    
#     _, xextnorm, _ = miescat(normwave[0], iscat, dsize[None,:], 1, rs, refindx_interp[0], theta[:ntheta], ntheta, nphase)
    return data_arr


@njit(fastmath=True, error_model='numpy')
def phase1(calpha, iscat, cons, icons=0, icont=None, ncont=None, vwave=None, pfunc = None, xmu = None):
    pi = np.pi
    calpha = min(max(calpha, -1.0), 1.0)
    if iscat == 0:
        p = 0.75 * (1.0 + calpha * calpha)
    elif iscat == 1:
        p = 1.0
    elif iscat == 2:
        f1 = cons[0]
        f2 = 1.0 - f1
        hg11 = 1.0 - cons[1] * cons[1]
        hg12 = 2.0 - hg11
        hg21 = 1.0 - cons[2] * cons[2]
        hg22 = 2.0 - hg21
        p = f1 * hg11 / np.sqrt(hg12 - 2.0 * cons[1] * calpha) ** 3 + f2 * hg21 / np.sqrt(hg22 - 2.0 * cons[2] * calpha) ** 3
    elif iscat == 3:
        cons[0] = 0.25 / pi
        p = 0.0
        xf = calpha * calpha
        for k in range(1, (icons + 1) // 2 + 1):
            n0 = 2 * k - 1
            n1 = 2 * k
            x0 = 1.0
            x1 = calpha
            pa0 = 0.0
            pa1 = 0.0
            for i in range(1, k + 1):
                pa0 += x0 * jcoef[i - 1, n0 - 1] / jdiv[k - 1]
                pa1 += x1 * jcoef[i - 1, n1 - 1] / jdiv[k - 1]
                x0 *= xf
                x1 *= xf
            p += cons[n0] * pa0 + cons[n1] * pa1
    elif iscat == 4:
        p = np.interp(calpha,xmu,pfunc)
        
    else:
        print('Error invalid scattering option.')
        return None
    p /= 4.0 * pi
    return p

@njit(fastmath=True, error_model='numpy')
def phasint2(nphi, ic, nmu, mu, iscat, cons, ncons, icont, ncont, vwave, pfunc, xmu, idump=0):
    pi = np.pi
    const = 1.0
#     if nphi <= nf:
#         print('WARNING: PHASINT2.F')
#         print('NPHI has been set less than NF, which will result in an unphysical oscillatory solution.')
#         return
    dphi = 2.0 * pi / nphi
    pplpl = np.zeros((nmu,nmu))
    pplmi = np.zeros((nmu,nmu))
    for j in range(0, nmu):
        for i in range(0, nmu):
            sthi = np.sqrt(1.0 - mu[i] * mu[i])
            sthj = np.sqrt(1.0 - mu[j] * mu[j])
            for k in range(0, nphi + 1):
                phi = k * dphi
                cpl = sthi * sthj * np.cos(phi) + mu[i] * mu[j]
                pl = phase1(cpl, iscat, cons, ncons, icont, ncont, vwave, pfunc, xmu)
                cmi = sthi * sthj * np.cos(phi) - mu[i] * mu[j]
                pm = phase1(cmi, iscat, cons, ncons, icont, ncont, vwave, pfunc, xmu)
                plx = pl * np.cos(ic * phi)
                pmx = pm * np.cos(ic * phi)
                wphi = dphi
                if k == 0 or k == nphi:
                    wphi = 0.5 * dphi
                if ic == 0:
                    wphi = wphi / (2.0 * pi)
                else:
                    wphi = wphi / pi
                pplpl[i, j] += wphi * plx
                pplmi[i, j] += wphi * pmx
    return (pplpl, pplmi)

@njit(fastmath=True, error_model='numpy')
def hansen(ic, ppl, pmi, maxp, wtmu, nmu, fc):
    pi = np.pi
    x1 = 2.0 * pi
    if nmu > maxp:
        print('Too many angles')
        return
    if ic == 0:
        rsum = np.zeros(nmu, dtype=float)
        for j in range(nmu):
            rsum[j] = np.sum(pmi[:, j] * wtmu) * x1
        niter = 0
        while True:
            niter += 1
            if niter > 10000:
                print('Normalization fails to converge')
                print('Leaving phase matrices uncorrected')
                return
            tsum = np.zeros(nmu, dtype=float)
            for j in range(nmu):
                tsum[j] = np.sum(ppl[:, j] * wtmu * fc[:, j]) * x1
            testj = np.abs(rsum + tsum - 1.0)
            test = np.max(testj)
            if test < 1e-14:
                break
            for j in range(nmu):
                xj = (1.0 - rsum[j]) / tsum[j]
                for i in range(j + 1):
                    xi = (1.0 - rsum[i]) / tsum[i]
                    fc[i, j] = 0.5 * (fc[i, j] * xj + fc[j, i] * xi)
                    fc[j, i] = fc[i, j]
    ppl *= fc
    return (ppl, fc)

@njit(fastmath=True)
def calc_pmat6(ic, mu, wtmu, nmu, iscat, cons8, ncons, norm, icont, ncont, vwave, nphi, fc, pfunc, xmu):
    pi = np.pi

    pplpl, pplmi = phasint2(nphi, ic, nmu, mu, iscat, cons8, ncons, icont, ncont, vwave, pfunc, xmu)
#     pplpl = ptpl[icont]
#     pplmi = ptmi[icont]
    if norm == 1:
        pplpl, fc = hansen(ic, pplpl, pplmi, MAXMU, wtmu, nmu, fc)
    elif norm == 2:
        idiag = 1
        if idiag > 0:
            print('CALC_PMAT6. NORM=2 Option disabled')
    return (pplpl, pplmi, fc)

@njit(fastmath=True)
def madd(const, am1, am2, n1, n2):
    ans = am1[:n1, :n2] + const * am2[:n1, :n2]
    return ans




# @njit(fastmath=True)
# def add_old(r1, t1, j1, e, nmu,ic, acom, ccom):

#     rsq = r1@r1
#     if r1[0,0]>0.001:
#         acom = np.linalg.solve(e - rsq, e)
#     else: 
#         acom = e + rsq
    
#     ccom = t1 @ acom
#     rans = ccom @ r1 @ t1 + r1
#     tans = ccom @ t1
    
#     if ic==0:
#         jcom = r1 @ j1 + j1
#         jans = ccom @ jcom + j1
#     else:
#         jans = j1

#     return rans, tans, jans
# @njit(fastmath=True)
# def elemult(mat1, mat2):
#     n = mat1.shape[0]
#     result = np.zeros((n, n))
#     for j in range(n):
#         for l in range(n):
#             for k in range(n):
#                 result[j, k] += mat1[j, l] * mat2[l, k]
#     return result

@njit(fastmath=True)
def elemult(mat1, mat2, result, n):
    for j in range(n):
        for l in range(n):
            for k in range(n):
                result[j, l] += mat1[j, k] * mat2[k, l]
                
@njit(fastmath=True)
def elemultvec(matrix, vector, result, n):
    for i in range(n):
        for j in range(n):
            result[i, 0] += matrix[i, j] * vector[j,0]


@njit(fastmath=True)
def add(r1, t1, j1, e, nmu,ic):
    rsq = np.zeros_like(r1)
    bcom = np.zeros_like(r1)
    ccom = np.zeros_like(r1)
    rans = np.zeros_like(r1)
    tans = np.zeros_like(r1)
    jcom = np.zeros_like(j1)
    jans = np.zeros_like(j1)
    
    elemult(r1,r1,rsq,nmu)
    if rsq[-1,-1]>0.01: # Spectral radius: Neumann series + Gershgorin circle theorem (bit dodgy)
        acom = np.linalg.solve(e - rsq, e)
    else: 
        acom = e + rsq
    elemult(t1, acom, ccom,nmu)
    elemult(ccom, r1, bcom,nmu)
    elemult(bcom, t1, rans,nmu)
    rans += r1
    elemult(ccom, t1,tans,nmu)
    
    if ic==0:
        elemultvec(r1, j1, jcom,nmu)
        jcom += j1
        elemultvec(ccom, jcom, jans,nmu)
        jans += j1
    else:
        jans = j1

    return rans, tans, jans


@njit(fastmath=True)
def numba_diagonal(arr):
    rows, cols = arr.shape
    length = min(rows, cols)
    diagonal = np.empty(length, arr.dtype)
    for i in range(length):
        diagonal[i] = arr[i, i]
    return diagonal

@njit(fastmath=True)
def numba_fill_diagonal(arr, vec):
    for i in range(len(vec)):
        arr[i, i] = vec[i]
        
@njit(fastmath = True)
def numba_sum_diagonal(arr):
    sum_ = 0
    for i in range(arr.shape[0]):
        sum_ += arr[i,i]
    return sum_

@njit(fastmath = True, error_model='numpy')
def double1(ic,l,nmu,jdim,cc,pplpl,pplmi,omega, mu,taut,bc,xfac,mminv,e,raman = False): #NEED TO ADD RAMAN
#     global xfac
    
    ipow0 = 12
#     if jdim!=MAXMU:
#         print('DOUBLE1: DIMENSION ERROR')
#         return

#     sum_val = np.sum(cc)
#     dsum = abs(sum_val - 1.0)
#     cc_diag = cc
#     if ic == 0:
#         for j in range(nmu):
#             sum_val = np.sum((pplpl[:, j] + pplmi[:, j]) * cc_diag)
#             dsum = np.abs(sum_val * 2.0 * np.pi - 1.0)
#             if dsum > 0.02:
#                 if True:
#                     print('Double1: Warning - Sum of phase function <> 1')
# #                     print(f'IC, L = {ic}, L')  # Assuming L is defined elsewhere
# #                     print(f'J, SUM = {j}, {sum_val * 2.0 * np.pi}')
# #                     print(sum_val * 2.0 * np.pi)
                    
# #                     print('PPLPL')
# #                     print(pplpl[:, j])
# #                     print('PPLMI')
# #                     print(pplmi[:, j])
# #                     print('Doing a brutal renormalisation')
                
#                 xfac = 1.0 / (2.0 * np.pi * sum_val)
#                 pplpl[:, j] *= xfac
#                 pplmi[:, j] *= xfac

    con = omega * np.pi
    del01 = 0.0
    if ic == 0:
        del01 = 1.0
    
    con *= (1.0 + del01)
    gplpl = mminv*(e-con*pplpl*cc)
    gplmi = mminv*(con*cc)*pplmi
    
    nn = int(np.log2(taut) + ipow0)
    xfac = 1.0 / (2.0 ** nn) if nn >= 1 else 1.0
    tau0 = taut * xfac
    
    # Computation of R, T and J for initial layer
    t1 = e - tau0 * gplpl.transpose()
    r1 = tau0 * gplmi.transpose()

    if ic == 0:
        j1 = (1.0 - omega) * bc * tau0 * mminv
    
    else:
        j1 = np.zeros((nmu, 1))
        
    
    if nn < 1:
        return r1, t1, j1
    
    for n in range(nn):
        r1, t1, j1 = add(r1, t1, j1, e, nmu, ic)
   
 #     print(repr(t1),flush=True)
    return r1, t1, j1

@njit(fastmath = True)
def addp(r1, t1, j1, iscat1,e, rsub, tsub, jsub, jdim, nmu): #needs checking

    if iscat1 == 1:
        rsq = np.zeros_like(r1)
        bcom = np.zeros_like(r1)
        ccom = np.zeros_like(r1)
        rans = np.zeros_like(r1)
        tans = np.zeros_like(r1)
        jcom = np.zeros_like(j1)
        jans = np.zeros_like(j1)
        
        
        # Second layer is scattering
        elemult(rsub,r1,rsq,nmu)
        if rsq[-1,-1]>0.01: # Spectral radius: Neumann series + Gershgorin circle theorem (bit dodgy)
            acom = np.linalg.solve(e - rsq, e)
            elemult(t1,acom,ccom,nmu)
        else:
            acom = e+rsq
            elemult(t1,acom,ccom,nmu)
        elemult(ccom,rsub,rans,nmu)
        elemult(rans,t1,bcom,nmu)
        rans = r1 + bcom
        elemult(ccom,tsub,tans,nmu)
        elemultvec(rsub,j1,jcom,nmu)
        jcom += jsub
        elemultvec(ccom,jcom,jans,nmu)
        jans += j1
    else:
        # Second layer is non-scattering
        jcom = np.zeros_like(j1)
        tans = np.zeros((nmu,nmu))
        rans = np.zeros((nmu,nmu))
        jans = np.zeros((nmu,1))
        
        elemultvec(rsub,j1,jcom,nmu)
        jcom += jsub
        
        
        for i in range(nmu):
            ta = t1[i, i]
            for j in range(nmu):
                tb = t1[j, j]
                tans[i, j] = tsub[i, j] * ta
                rans[i, j] = rsub[i, j] * ta * tb
            jans[i, 0] = j1[i, 0] + ta * jcom[i, 0]
    return rans, tans, jans

@njit
def iup(ra, ta, ja, rb, tb, jb, u0pl, utmi):
    e = np.identity(len(ra))
    acom = np.dot(rb, ra)
    bcom = e - acom
    bcom = np.linalg.inv(bcom)
    xcom = np.dot(tb, utmi)
    acom = np.dot(rb, ta)
    ycom = np.dot(acom, u0pl)
    xcom += ycom
    ycom = np.dot(rb, ja)
    umi = xcom + ycom
    xcom = umi + jb
    umi = np.dot(bcom, xcom)
    return umi

@njit
def idown(ra, ta, ja, rb, tb, jb, u0pl, utmi):
    e = np.identity(len(ra))
    acom = np.dot(ra, rb)
    bcom = e - acom
    bcom = np.linalg.inv(bcom)
    xcom = np.dot(ta, u0pl)
    acom = np.dot(ra, tb)
    ycom = np.dot(acom, u0pl)
    xcom += ycom
    ycom = np.dot(ra, jb)
    upl = xcom + ycom
    xcom = upl + ja
    upl = np.dot(bcom, xcom)
    return upl

@njit(fastmath = True,parallel=False, cache = False, error_model='numpy')
def scloud11wave(phasarr, radg, sol_ang, emiss_ang, solar, aphi, lowbc, galb, mu1, wt1, nf,
                vwaves, bnu, tau, tauray,omegas, nphi,iray, lfrac, imie=0):
    
    """
    Calculate spectrum using the adding-doubling method.

    Parameters
    ----------
    phasarr(NMODES,NWAVE,3,NPHAS):
        If imie = 0, contains fitted HG-phase function parameters f, g1, g2 and NPHAS = 1
        If imie = 1, contains phase functions (phasarr[:,:,0,:]) and corresponding angle grid (phasarr[:,:,1,:])
    radg(NWAVE,NMU): 
        Incident intensity at the bottom of the atm
    sol_ang:
        Solar zenith angle (degrees)
    emiss_ang:
        Emission zenith angle (degrees)
    solar(NWAVE):        
        Incident solar flux at the top of the atm
    aphi:
        Azimuth angle between Sun and observer
    lowbc:
        Lower boundary condition: 0 = thermal, 1 = Lambert reflection
    galb(NWAVE):
        Ground albedo at the bottom
    mu1(NMU):  
        Zenith angle point quadrature
    wt1(NMU):  
        Zenith angle point quadrature
    nf:
        Maximum number of terms in azimuth Fourier expansion 
    vwaves(NWAVE):
        Input wavelengths.
    bnu(NWAVE,NLAY):
        Mean Planck function in each layer
    tau(NWAVE,NLAY):
        Total optical thickness of each layer
    tauray(NWAVE,NLAY):
        Rayleigh optical thickness of each layer
    omegas(NWAVE,NLAY):
        Single scattering albedo of each layer
    nphi:
        Number of azimuth integration ordinates
    iray:
        Flag for using rayleigh scattering
    lfrac(NCONT,NLAY):
        Fraction of scattering contributed by each type in each layer
    imie:
        Flag for phasarr behaviour
    
    Returns
    -------
    rad(NWAVE):
        Upwards radiance at each wavelength
    """
        
    nwave = len(vwaves)
    nmu = len(mu1)
    nlay = tau.shape[1]
    ncont = phasarr.shape[0]
    
    ltot = nlay
    lt1 = ltot
    nf = nf + 1
    
    idump = 0
    fours = 0
    xfac = 0.
    xfac = np.sum(mu1*wt1)
    xfac = 0.5/xfac    
    
    # Reset the order of angles
    mu = mu1[::-1]
    wtmu = wt1[::-1]

    pi = np.pi
    
    yx = np.zeros((4))
    u0pl = np.zeros((nmu,1))
    utmi = np.zeros((nmu,1))

    umi = np.zeros((nlay,nmu))
    upl = np.zeros((nlay,nmu))

    ppln = np.zeros((ncont, nmu, nmu))
    pmin = np.zeros((ncont, nmu, nmu))
    pplr = np.zeros((1, nmu, nmu))
    pmir = np.zeros((1, nmu, nmu))
    rl = np.zeros((nmu,nmu))
    tl = np.zeros((nmu,nmu))
    jl = np.zeros((nmu,1))

    rbase = np.zeros((nmu,nmu))
    tbase = np.zeros((nmu,nmu))
    jbase = np.zeros((nmu,1))

    iscl = np.zeros((nwave))

    uplf = np.zeros((nlay+lowbc,nmu))
    umif = np.zeros((nlay+lowbc,nmu))

    pplpls = np.zeros((nwave,nmu, nmu))
    pplmis = np.zeros((nwave,nmu, nmu))
    
    # Set up constant matrices
    e = np.identity(nmu)
    mm = np.zeros((nmu, nmu))
    numba_fill_diagonal(mm, mu[:nmu])
    mminv = 1/mu
    mminv = mminv[:,None]
    cc = wtmu
    cc = cc[None,:]
    ccinv = np.zeros((nmu, nmu))
    numba_fill_diagonal(ccinv, 1/wtmu[:nmu])
    rad = np.zeros(nwave)
    jdim = 21
    
    
    if sol_ang > 90.0:
        zmu0 = np.cos(np.radians(180 - sol_ang))
        solar1 = solar*0.0
    else:
        zmu0 = np.cos(np.radians(sol_ang))
        solar1 = solar

    zmu = np.cos(np.radians(emiss_ang))

    isol = 1
    for j in range(nmu-1):
        if zmu0 <= mu[j] and zmu0 > mu[j+1]:
            isol = j+1
    if zmu0 <= mu[nmu-1]:
        isol = nmu - 1

    iemm = 1
    for j in range(nmu-1):
        if zmu <= mu[j] and zmu > mu[j+1]:
            iemm = j+1
    if zmu <= mu[nmu-1]:
        iemm = nmu - 1

    fsol = (mu[isol-1] - zmu0) / (mu[isol-1] - mu[isol])
    femm = (mu[iemm-1] - zmu) / (mu[iemm-1] - mu[iemm])
    
    t = femm
    u = fsol

    radg = radg[:,::-1]
    
    fc = np.ones((ncont+1,nmu,nmu))
    
    for widx in prange(len(vwaves)):
        converged = False
        conv1 = False
        vwave = vwaves[widx]
        defconv = 1e-3
        for ic in range(nf):
            ppln*=0
            pmin*=0 
            pplr*=0
            pmir*=0
            
            for j1 in range(ncont):
                if imie == 0:
                    f, g1, g2 = phasarr[j1, widx, :, 0]  
                    iscat = 2
                    ncons = 3
                    cons8 = np.array([f, g1, g2])
                    norm = 1
                    pfunc = cons8
                    xmu = cons8
                    
                    pplpl, pplmi, fc[j1] = calc_pmat6(ic, mu, wtmu, nmu, iscat, cons8, ncons, 
                                              norm, j1, ncont, vwave, nphi, fc[j1], pfunc, xmu)
                    # Transfer matrices to those for each scattering particle
                    ppln[j1] = pplpl
                    pmin[j1] = pplmi
                    
                else:
                    pfunc = phasarr[j1, widx, 0, :]
                    xmu   = phasarr[j1, widx, 1, :]
                    iscat = 4
                    ncons = 0
                    cons8 = pfunc
                    norm = 1
                    pplpl, pplmi, fc[j1] = calc_pmat6(ic, mu, wtmu, nmu, iscat, cons8, ncons, 
                                              norm, j1, ncont, vwave, nphi, fc[j1], pfunc, xmu)
                    # Transfer matrices to those for each scattering particle
                    ppln[j1] = pplpl
                    pmin[j1] = pplmi
                
                
            if iray > 0:
                iscat = 0
                ncons = 0
                pplpl, pplmi, fc[ncont] = calc_pmat6(ic, mu, wtmu, nmu, iscat, cons8, ncons, 
                                            1, ncont, ncont, vwave, nphi, fc[ncont], pfunc, xmu)

                # Transfer matrices to storage
                pplr[0] = pplpl
                pmir[0] = pplmi


            for l in range(0,ltot):
                k = ltot - l - 1 
                iscl[widx] = 0

                taut = tau[widx,k]
                bc = bnu[widx,k]
                omega = omegas[widx,k]
                if omega < 0:
                    omega = 0.0
                if omega > 1:
                    omega = 1.0


                tauscat = taut*omega
                taur = tauray[widx,k]
                tauscat = tauscat-taur


                if tauscat < 0:
                    tauscat = 0.0   
                omega = (tauscat+taur)/taut

                if l == 0 and lowbc == 1:
                    jl[:,0] = (1-galb[widx])*radg[widx]
                    if ic == 0:
                        tl *= 0.0
                        for j in range(nmu):
                            rl[j,:] = 2*galb[widx]*mu[j]*wtmu[j] 
                            rl[:,:]*= xfac
                    else:
                        tl *= 0.0
                        rl *= 0.0
                    jbase = jl
                    rbase = rl
                    tbase = tl

                if taut == 0:
                    rl *= 0.0
                    tl *= 0.0
                    jl *= 0.0
                    for i in range(nmu):
                        tl[i,i] = 1.0

                elif omega == 0:

                    rl *= 0.0
                    tl *= 0.0
                    for i1 in range(nmu):
                        tex = -mminv[i1,0]*taut
                        if tex > -200.0:
                            tl[i1,i1] = np.exp(tex)
                        else:
                            tl[i1,i1] = 0.0

                        jl[i1,0] = bc*(1.0 - tl[i1,i1])

                else:
                      # ADD IN LFRAC SUPPORT
                    pplpl = (taur/(tauscat+taur))*pplr[0]
                    pplmi = (taur/(tauscat+taur))*pmir[0]


                    for j1 in range(ncont):
                        pplpl += tauscat/(tauscat+taur)*ppln[j1]*lfrac[widx,j1,k] 
                        pplmi += tauscat/(tauscat+taur)*pmin[j1]*lfrac[widx,j1,k] 

                    iscl[widx] = 1
                    rl, tl, jl = double1(ic,l,nmu,jdim,cc,pplpl,pplmi,omega,mu1,taut,bc,xfac, mminv,e)

                if l == 0 and lowbc == 0:
                    jbase = jl
                    rbase = rl
                    tbase = tl
                else:
                    rbase, tbase, jbase = addp(rl, tl, jl, iscl[widx],e,
                                                                      rbase, tbase, jbase,
                                                                      jdim, nmu)


            if ic != 0:
                jbase *= 0.0


            for j in range(nmu):
                u0pl[j] = 0.0
                if ic == 0:
                    utmi[j] = radg[widx,j]
                else:
                    utmi[j] = 0.0

            ico = 0

            for imu0 in range(isol-1, isol+1):  # Adjust for zero-indexing
                u0pl[imu0] = solar1[widx] / (2.0 * np.pi * wtmu[imu0])
                acom = rbase.transpose() @ u0pl
                bcom = tbase @ utmi
                acom += bcom
                umi = acom + jbase
                for imu in range(iemm-1, iemm+1): 
                    yx[ico] = umi[imu,0]
                    ico += 1
                    u0pl[imu0] = 0.0 

            drad = ((1-t)*(1-u)*yx[0] + t*(1-u)*yx[1] + t*u*yx[3] + (1-t)*u*yx[2]) * np.cos(ic*aphi * np.pi / 180.0)

            if ic > 0:
                drad *= 2

            rad[widx] += drad
            conv = np.abs(drad / rad[widx])

            if conv < defconv and conv1:
                break
                
            if conv < defconv:
                conv1 = True
            else:
                conv1 = False
    
    return rad

def ramanjsource(fmean, vv, lambda0, dens, fpraman, jraman, vram0, vramst, nraman, h2abund, iraman, kwt):
    lambda0 = 10000.0 / vv
    ilambda0 = int((lambda0 - vram0) / vramst)
    specs0, specs1, specq = ramanxsec(lambda0)
    transitions = [(DNUS0, specs0), (DNUS1, specs1), (DNUQ, specq)]
    for dnu, specs in transitions:
        lambda_val = 10000.0 / (vv - dnu)
        ilambda = int((lambda_val - vram0) / vramst)
        if ilambda == ilambda0:
            print('Error in ramanjsource')
            print('Raman source function computed to lie within current bin.')
            print('Need to use JRAMAN array with smaller step-size')
            return
        if 1 <= ilambda < nraman:
            for i in range(len(fmean)):
                fpara = fpraman[i] if dnu != DNUQ else 1
                spec = fpara * specs
                sig = fmean[i] * h2abund[i] * spec / np.pi
                sig = (vv - dnu) * sig / vv
                factor = sig * kwt
                if abs(ilambda - ilambda0) > 1 and ilambda > 1:
                    jraman[ilambda - 1, i] += 0.25 * factor
                    jraman[ilambda, i] += 0.5 * factor
                    jraman[ilambda + 1, i] += 0.25 * factor
                else:
                    jraman[ilambda, i] += factor
    return jraman


