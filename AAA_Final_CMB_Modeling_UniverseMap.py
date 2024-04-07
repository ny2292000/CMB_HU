from time import sleep

# from lib4 import *
from lib4 import *
import numpy as np
from scipy.optimize import curve_fit
mypath = "./PG_data"
x0 = [4.93231069, 4.97130803, 0.85524497] # interrupted optimized value 03/27/2024

todo =[
#          Color.FINDNEIGHBORHOOD,
         Color.EVALUATE_DF_AT_POSITION,
#          Color.MINIMIZEPOSITION,
         Color.CREATE_HIGH_RESOL_BACKGROUNDPLOT,
         Color.OPTIMIZE_SPECTRUM,
         Color.OPTIMIZE_SMICA_BACKGROUND,
         Color.CREATE_GAUSSIAN_BACKGROUND,
         Color.CREATE_HISTOGRAM,
         Color.CHARACTERIZE_SPECTRUM, 
         Color.CREATEMAPOFUNIVERSE,
         # Color.FINDBESTFORKRANGE,
#          Color.CREATE_VARIABLE_R_GIF,
         Color.WORK_86_128,
]



if Color.FINDNEIGHBORHOOD in todo:
    # these three indices are related to position within the hyperspherical hypersurface.  Don't confuse them with the
    # quantum numbers k,l,m
    x0 = [0.0, 0.0, 0.0]
    (lambda_k, lambda_l, lambda_m) = x0
    # Create karray
    nside3D = 48
    bandwidth = 48
    karray = list(np.arange(2, 10))
    print(len(karray))
    kmax = max(karray)
    #################################
    myHyper = HYPER(nside3D, sigma_smica, planck_IQU_SMICA, karray,mypath,
                        lambda_k, lambda_l, lambda_m, loadpriorG=False,
                        savePG=False, bandwidth=bandwidth)

    #################################################################
    # these three indices are related to position within the hyperspherical hypersurface.  Don't confuse them with the
    # quantum numbers k,l,m
    errarray = []
    olderr = 1110.0
    results = 0.0
    x00 = []
    nside3D = 48
    bandwidth = 48
    n = 7
    myHyper.change_SMICA_resolution(nside3D, doit=False, bandwidth=bandwidth)
    for lk in np.linspace(0, 2 * np.pi, n):
        for ll in np.linspace(0, 2 * np.pi, n):
            for lm in np.linspace(0, 2 * np.pi, n):
                start_time = time()
                try:
                    myHyper.change_HSH_center(lk, ll, lm, karray, nside3D, loadpriorG=False, doit=True,
                                              savePG=False)
                    results, fcolors, err = myHyper.project4D3d_0(karray)
                    myHyper.plotNewMap(fcolors,kmax, err, filename=None, title=None, plotme=False, save=True)
                    if olderr > err:
                        olderr = err
                        err0 = np.round(err, 3)
                        filename = "./img1/Bestf_{}_{}_{}__{}_{}_{}.png".format(myHyper.kmax, myHyper.nside3D,
                                                                                err0,
                                                                                chg2ang(lk),
                                                                                chg2ang(ll),
                                                                                chg2ang(lm))
                        myHyper.plotNewMap(fcolors, err, filename=filename, title=None, plotme=True, save=True)
                        x00.append((lk, ll, lm, err))
                        np.save("./img1/x0_{}_{}.npy".format(kmax, nside3D), x00[-1:], allow_pickle=True)
                        print(lk, ll, lm, err)
                except Exception as aa:
                    print('Error with getting map for: {}_{}_{}'.format(lk, ll, lm))
                stop_time = time()
    myHyper.creategiff(kmax, nside3D, mypath="./img1", prefix="aitoff_", filename="CMB")
    sleep(10)
    myHyper.creategiff(kmax, nside3D, mypath="./img1", prefix="Bestf", filename="BEST")
    sleep(10)
    myHyper.cleanup(mypath="./img1", prefix="aitoff_")
    # myHyper.cleanup(mypath="./img1", prefix="Bestf")
    print("Best Position = ", x00[-1:])
    bestposition = "./img1/x0_{}_{}.npy".format(kmax, nside3D)
    np.save("./img1/x0_{}_{}.npy".format(kmax, nside3D), x00[-1:], allow_pickle=True)
    np.save("./img1/x00_{}_{}.npy".format(kmax, nside3D), x00, allow_pickle=True)

if Color.MINIMIZEPOSITION in todo:
    x0 = [4.93231069, 4.97130803, 0.85524497] # interrupted optimized value 03/27/2024
    
    nside3D = 64
    bandwidth = 64
    (lambda_k, lambda_l, lambda_m) = x0
    # Create karray
    karray = list(np.arange(2, 30))
    print(len(karray))
    kmax = max(karray)
    #################################
    myHyper = HYPER(nside3D, sigma_smica, planck_IQU_SMICA, karray,mypath,
                        lambda_k, lambda_l, lambda_m, loadpriorG=False, savePG=False, bandwidth=bandwidth)
    print(x0)
    myHyper.change_SMICA_resolution(nside3D, doit=False, bandwidth=bandwidth)
    x0 = minimize(myHyper.calcError, x0, args=(karray,nside3D), method='nelder-mead',
                  options={'xatol': 1e-4, 'disp': True})
    np.save("./img1/x0_{}_{}.npy".format(kmax, nside3D), x0.x, allow_pickle=True)
    x0 = x0.x
    (lk, ll, lm) = [x+1 for x in x0]
    print("minimized at {}".format(x0))
    filename = "./img1/Best_{}_{}_{}__{}_{}_{}.png".format(kmax, nside3D, bandwidth, chg2ang(lk), chg2ang(ll),
                                                           chg2ang(lm))
    title = "Best_{}_{}_{}__{}_{}_{}".format(kmax, nside3D, bandwidth, chg2ang(lk), chg2ang(ll), chg2ang(lm))
    myHyper.change_HSH_center(lk, ll, lm, karray, nside3D, loadpriorG=False, doit=True, savePG=True)
    results, newmap, err = myHyper.project4D3d_0(karray)
    np.save("./img1/results_{}_{}_{}.npy".format(kmax, nside3D, bandwidth), results, allow_pickle=True)
    (lk, ll, lm) = [x+1 for x in x0]
    filename = "./img1/BestMaximized_{}_{}_{}__{}_{}_{}.png".format(kmax, nside3D, bandwidth, chg2ang(lk), chg2ang(ll),
                                                           chg2ang(lm))
    title = "BestMaximized_{}_{}_{}__{}_{}_{}".format(kmax, nside3D, bandwidth, chg2ang(lk), chg2ang(ll), chg2ang(lm))
    myHyper.plotNewMap(newmap, err, filename=filename, title=title, plotme=True, save=True)
    print("maximized at ", x0)

if Color.EVALUATE_DF_AT_POSITION in todo:
    myHyper = None
    # Earth Position
    x0 = [4.93231069, 4.97130803, 0.85524497] # interrupted optimized value 03/27/2024
    y0 = [np.round((xx + 1) / np.pi * 180, 2) for xx in x0]
    print("Earth Position", y0)
    (lk, ll, lm) = x0

    print("evaluated at {}".format(x0))
    # Create karray
    karray = list(np.arange(2, 48))

    nside3D = 64
    bandwidth = 64
    print(len(karray))
    kmax = max(karray)
    #################################
    myHyper = HYPER(nside3D, sigma_smica, planck_IQU_SMICA, karray ,mypath,
                        lk, ll, lm, loadpriorG=False, savePG=False, bandwidth=bandwidth, longG=False)
    #################################################################
    results, newmap0, err = myHyper.project4D3d_0(karray)
    np.save("./results.npy", results)
    #########################################################################
    (lk, ll, lm) = [x+1 for x in x0]
    filename = "./img1/Best_{}_{}_{}__{}_{}_{}.png".format(kmax, nside3D, bandwidth, chg2ang(lk), chg2ang(ll),
                                                           chg2ang(lm))
    title = "Best_{}_{}_{}__{}_{}_{}".format(kmax, nside3D, bandwidth, chg2ang(lk), chg2ang(ll), chg2ang(lm))
    myHyper.plotNewMap(newmap0, err, filename=filename, title=title, plotme=True, save=True)


if Color.CREATE_HIGH_RESOL_BACKGROUNDPLOT in todo:
    nside3D=1024 # 4096
    newmap0_lg = myHyper.change_resolution(newmap0.squeeze(), nside3D=nside3D, plotme=True, title= "BestFit", bandwidth=64)
    SMICA_lg = myHyper.change_resolution(myHyper.SMICA.squeeze(), nside3D=nside3D, plotme=True,title= "SMICA", bandwidth=nside3D)
    # SMICA_lg = myHyper.SMICA.squeeze()
    # diffmap_lg = SMICA_lg - newmap0_lg
    # myHyper.plotNewMap(diffmap_lg, err, filename=None, title="Difference Map", plotme=True, save=False, nosigma=False)
    
    x01 = np.array([0.00161687, 0.31435231])
    x00 = minimize(newerr, x01, args=(SMICA_lg, newmap0_lg),
                   method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
    err = x00.fun
    xx0 = x00.x
    newmap0_lg =xx0[1]*newmap0_lg  #+ xx0[0]
    diffmap_lg = newmap0_lg- SMICA_lg
    (mu, sigma) = norm.fit(diffmap_lg)
    if sigma != 0.0:
        diffmap_lg = diffmap_lg / sigma * myHyper.sigma_smica           
    myHyper.plotNewMap(diffmap_lg, err, filename=None, title="Difference Map", plotme=True, save=False, nosigma=False)
    np.save("./img1/diffmap.npy", diffmap_lg )

if Color.CREATE_HISTOGRAM in todo:
    # Create Histogram
#     x01 = np.array([0.00161687, 0.31435231])
#     x00 = minimize(newerr, x01, args=(myHyper.SMICA_LR.squeeze(), newmap0.squeeze()),
#                    method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
#     err = x00.fun
#     xx0 = x00.x
#     newmap0_lg
#     newmap, diffmap = myHyper.optimizeNewMap(newmap0.squeeze(), myHyper.SMICA_LR.squeeze(),
#                                                                xx0=xx0,
#                                                                nside3D=nside3D, bandwidth=bandwidth, nosigma=True)
    myHyper.plotHistogram(newmap0_lg, nside3D, kmax, plotme=True)
    #################################################################
    #################################################################
    #################################################################
    #########################################################################     
        
    filename = "./img1/SingleBestOptimizedColor_{}_{}_{}__{}_{}_{}.png".format(kmax, nside3D, bandwidth, chg2ang(lk),
                                                                 chg2ang(ll), chg2ang(lm))
    title = "BestOptimizedColor_{}_{}_{}_{}__{}_{}_{}".format(kmax, nside3D, bandwidth, err, chg2ang(lk),
                                                chg2ang(ll), chg2ang(lm))

    myHyper.plotNewMap(newmap0_lg, err, filename=filename, title=title, plotme=True, save=True, nosigma=False)
    myHyper.plotNewMap(myHyper.SMICA_LR, err, filename=None, title="SMICA", plotme=True, save=False, nosigma=False)


if Color.CHARACTERIZE_SPECTRUM in todo:
    df = pd.DataFrame(np.concatenate([results, myHyper.df[:, 0:4]], axis=1),
                      columns=["coeff", "k", "l", "m", "std"])
    filename = "./PG_data/spectrumOptimum_{}_{}__{}_{}_{}".format(kmax, nside3D, chg2ang(lk), chg2ang(ll),
                                                                  chg2ang(lm))
    df.to_pickle(filename)

    df["CL"] = df["coeff"] ** 2
    df["abscoeff"] = df.coeff.abs()

    fcolors = df.coeff.values
    (mu, sigma) = norm.fit(fcolors)
    n, bins, patch = plt.hist(fcolors, 600, density=1, facecolor="r", alpha=0.25)
    y = norm.pdf(bins, mu, sigma)
    plt.plot(bins, y)
    plt.xlim(mu - 4 * sigma, mu + 4 * sigma)
    plt.xlabel("Amplitude")
    plt.ylabel("Frequency")
    plt.ylim(0, 0.04)
    plt.title("Amplitude Histogram \n HU Modeling of Planck SMICA Map")
    plt.savefig("./PG_data/AmplitudeHistogram_{}_{}.png", dpi=300)
    plt.show()

    numberofmodes = df.groupby(['k'])["abscoeff"].count()
    meanAmplitude = df.groupby(['k'])["abscoeff"].mean()
    meanEnergy = meanAmplitude ** 2 * meanAmplitude.index * numberofmodes
    stdd = df.groupby(['k'])["abscoeff"].std()
    one = np.ones(stdd.shape)
    valueOfK = stdd.index.values

    # calculation of Predicted Amplitudes
    predictedAmp = 45 / np.sqrt(valueOfK)
    # low pass filtering
    predictedAmp = np.exp(-0.01 * (valueOfK - 2)) * predictedAmp
    # Frequency dependent freezing phase accounting
    delta = 0.01
    predictedAmp = sinxx(valueOfK * delta) * predictedAmp
    plt.plot(numberofmodes, predictedAmp)
    plt.scatter(numberofmodes, meanAmplitude)
    plt.title("Predicted and Observed Mean Amplitudes \n under Energy Equipartition per mode k")
    plt.xlabel("Accessible Modes for given k")
    plt.ylabel("Hyperspherical Mode Amplitude per k")
    plt.ylim(0, 30)
    plt.xlim(0, 250)
    plt.savefig("./PG_data/PredicedtAmp.png", dpi=300)
    plt.show()

    plt.plot(numberofmodes, meanEnergy)
    plt.title("Mean Energy per k")
    plt.xlabel("Accessible Modes for given k")
    plt.ylabel("Mean Energy per k \n arbitrary units")
    plt.xlim(0, 250)
    # plt.ylim(0, 1E7)
    plt.savefig("./PG_data/MeanEnergy.png", dpi=300)
    plt.show()

    plt.scatter(numberofmodes, meanAmplitude)
    plt.title("Mean Amplitude per k")
    plt.xlabel("Accessible Modes for given k")
    plt.ylabel("Mean Amplitude per k")
    plt.ylim(0, 30)
    plt.xlim(0, 250)
    plt.savefig("./PG_data/MeanAmp.png", dpi=300)
    plt.show()

    plt.scatter(numberofmodes, stdd)
    plt.title("STD")
    plt.xlabel("Accessible Modes for given k")
    plt.ylabel("STD")
    plt.xlim(0, 250)
    plt.savefig("./PG_data/STD.png", dpi=300)
    plt.show()

    plt.plot(numberofmodes, meanAmplitude / stdd)
    plt.plot(numberofmodes, np.mean((meanAmplitude / stdd)[3:]) * one)
    plt.title("MeanAmplitude/std")
    plt.xlabel("Accessible Modes for given k")
    plt.ylabel("MeanAmplitude/std")
    plt.xlim(0, 250)
    # plt.ylim(0, 2)
    plt.savefig("./PG_data/PredicedtAmpPerSTD.png", dpi=300)
    plt.show()
    ################################################################
    ################################################################
    ################################################################
    ################################################################
    ################################################################
    ################################################################
    #################################################################


# if Color.CREATEMAPOFUNIVERSE in todo:
#     # rr = np.concatenate([np.linspace(0.01,0.99,99), np.linspace(0.999,1,10)])
#     rr = np.linspace(0.99999993,1,7)
#     def functionK(i,rr=rr,results=results, myHyper=myHyper):
#         r= rr[i]
#         myHyper.recalcXYZ(r)
#         _, fcolors=myHyper.getCMBxyz(i, results)
#         if r < 0.99:
#             r_round=np.round(r,2)
#         else:
#             r_round=r
#         filename = "./universeorthoview/radius_{}.png".format(r_round)
#         title = "Radius_{}".format(r_round)
#         myHyper.plotNewMap(fcolors.squeeze(), err, filename=filename, title=title, plotme=True, save=True)
#
#
#     do_multiprocessing(functionK, iterable=np.arange(len(rr)), number_of_concurrent_processes=10)
#



if Color.CREATEMAPOFUNIVERSE in todo:
    # Example of how to use the modified method:
    # Initialize your HYPER instance as `myHyper` and assume `results` are defined
    # results = np.load("./results.npy")
    # rr = np.concatenate([np.linspace(0.01,0.99,99), np.linspace(0.999,1,10)])
    # map3D, all_fcolors = myHyper.getUniverseMap(results, rr, save_images=True)

    rr = np.concatenate([np.linspace(0.01,0.99,99), np.linspace(0.999,1,10)])
    map3D, all_fcolors = myHyper.getUniverseMap(results, rr)
    for i, r in enumerate(rr): # np.linspace(0.01,1,100):
        if r < 0.99:
            r_round=np.round(r,2)
        else:
            r_round=r
        filename = "./universeorthoview/radius_{}.png".format(r_round)
        filenameG = "./universeorthoviewdual/globe_radius_{}.png".format(r_round)
        title = "Radius_{}".format(r_round)
        myHyper.plotNewMap(all_fcolors[i], err, filename=filename, title=title, plotme=True, save=True)
        stdd=all_fcolors[i].std()
        hp.orthview(all_fcolors[i].squeeze(), min=-3 * stdd, max=3 * stdd, half_sky=False, nest=False,
                    cmap='magma',
                    title = title, unit = r'$\Delta$T (mK)')
        plt.savefig(filenameG)
        plt.show()

if Color.CREATEMAPOFUNIVERSE in todo:
    def creategiff(kmax, nside3D, mypath="./img1", prefix="aitoff_", filename="CMB"):
        giffiles = myHyper.getSpectralFiles(mypath)
        gifnumbers = [x.replace(prefix,"").replace(".png", "").replace(mypath,"") for x in giffiles if prefix in x]
        gifnumbers = list(set(gifnumbers))
        gifnumbers = np.sort(gifnumbers)
        images = []
        for fff in gifnumbers:
            ff =  mypath + prefix + fff + ".png"
            print(ff)

        for fff in gifnumbers:
            ff =  mypath + prefix + fff + ".png"
            print(ff)
            with Image.open(ff).convert("RGB") as img:
                images.append(img)
        fname = os.path.join(mypath, filename) + '_{}_{}.gif'.format(kmax, nside3D)
        images[0].save(fname, save_all=True, append_images=images[1:], optimize=False, duration=100000, loop=1)


    creategiff(kmax, nside3D, mypath="./universeorthoview/", prefix="radius_", filename="CMB_Universe_Tomography")


if Color.OPTIMIZE_SMICA_BACKGROUND in todo:
    beam_arc_min=10
    white_noise = np.ma.asarray(np.random.normal(0, sigma_smica, 12*nside3D** 2))
    cl_WHITE_NOISE, dl_WHITE_NOISE, ell = get_dl(white_noise, nside=nside3D, beam_arc_min=1)
    cl_SMICA, dl_SMICA, ell = get_dl(diffmap_lg.squeeze(), nside=nside3D, beam_arc_min=beam_arc_min)
    def yy_func(x, ell):
        return x[0] + x[1] * np.exp(ell * x[2] + ell**2 *x[3])
    
    def olderr(x, dl_SMICA, ell):
        err = dl_SMICA - yy_func(x,ell)
        err = np.sum(err * err)
        print(err, x)
        return err


    dl_SMICA1 = xx0[1]*dl_SMICA+xx0[0]
    x01 = np.array([ 0.00618878,  0.00087167,  0.00037092, -0.002705  , -0.00031672])
    aa = 2000
    bb= 2800
    x = ell[aa:bb].squeeze()
    y = dl_SMICA1[aa:bb].squeeze()
    x00 = minimize(olderr, x01, args=(y, x),
                   method='nelder-mead', options={'xatol': 1e-6, 'disp': True})
    err = x00.fun
    xx0 = x00.x
    yy = yy_func(xx0, ell) 
    dl_SMICA_Clean = dl_SMICA1 - yy
    # a0 = pars[0]  # Amplitude
    # a1 = pars[1]  # center
    # a2 = pars[2]  # std
    # parguess = [326.07961712,   240.5495938,    130.86283279,
    #             185.2806624, 511.71738065,   106.82439558,
    #             654.09021179,   778.65904957, 195.83549305,
    #             50.093336,    1114.17415044,   102.05325759,
    #             105.70709484,  1272.23029976,   184.05430212,
    #             -1185.72051889, 666.00139292,   367.6208938 ]

    mygauss = fitClass()
    mygauss.n=5
    parguess = np.array([1.13141056e+02, 2.54610864e+02, 8.41771175e+01, 4.92402013e-03,
                         9.80458369e+02, 5.95410382e+02, 8.96299871e+01, 7.51841770e-03,
                         2.60752925e+01, 8.22194747e+02, 9.18280463e+01, 7.75142569e-04,
                         4.60311565e+01, 1.14621194e+03, 8.47451004e+01, 2.12699957e-03,
                         4.00537143e-02, 1.41213361e+03, 6.14342868e+01, -2.42490682e-03])
    plt.plot(ell, mygauss.six_peaks(ell, *parguess), 'r-')
    plt.show()
    np.save("./PG_data/ell.npy", ell)
    np.save("./PG_data/dl_SMICA_Clean.npy", dl_SMICA_Clean)
    popt, _ = curve_fit(mygauss.six_peaks, ell[0:2000], dl_SMICA_Clean[0:2000], parguess, maxfev = 420000)
    print(popt)
    parguess = np.array(popt).reshape([5, 4])
    gamma= popt[-1:][0]
    # parguess = parguess[parguess[:, 1].argsort()]
    centers = np.array([parguess[x, 1] for x in np.arange(5)])
    freqs = centers[1::] - centers[0:-1:]
    amplt = np.array([parguess[x, 0] for x in np.arange(5)])
#     fitting1 = np.polyfit(centers, np.log(amplt), 1)
#     yy = np.exp(fitting1[1]) * np.exp(fitting1[0] * ell)
#     # First peak is at pi/2=l*Delta => Delta=pi/2/l
#     delta = np.pi / 2 / centers[0]
#     gamma1 = fitting1[0] / delta

# #     ################################################################
# #     ################################################################
#     fig, axis1 = plt.subplots()
#     axis1.plot(ell, dl_SMICA_Clean)
#     for i in np.arange(5):
#         axis1.plot(ell, mygauss.fitfun(ell, *parguess[i, :]), 'r-')
#     axis1.plot(ell, mygauss.six_peaks(ell, *popt), 'r-')
#     axis1.set_xlim([0, 2000])
#     axis1.set_ylim([0, None])
#     axis1.set_xlabel('Spherical Harmonic L')
#     axis1.set_ylabel('Intensity (arb. units)')
#     axis1.set_title("Dissipation on High-Fequency CMB Power Spectrum \n delta ={}  Gamma={}".format('%.4E', '%.4E') %
#               (delta, gamma1))
#     plt.legend(['Power Spectrum', 'Fitted Data'])
#     plt.savefig('./img1/HighFreqFittedPowerSpectrum.png')
#     plt.show()
# #     ################################################################
# #     ################################################################
#     fig, axis1 = plt.subplots()
#     fitting1 = np.polyfit(centers, np.log(amplt), 1)
#     yy = np.exp(fitting1[1]) * np.exp(fitting1[0] * ell)
#     # First peak is at pi/2=l*Delta => Delta=pi/2/l
#     plt.scatter(centers, amplt)
#     plt.plot(ell, yy)
#     plt.xlim([0, 2000])
#     # plt.ylim([-200, 2000])
#     plt.xlabel('Spherical Harmonic L')
#     plt.ylabel('Intensity (arb. units)')
#     plt.title("Dissipation on High-Fequency CMB Power Spectrum \n delta ={}  Gamma={}".format('%.4E', '%.4E') % (
#         delta, gamma1))
#     plt.legend(['Power Spectrum', 'Fitted Data'])
#     # plt.savefig('./img1/HighFreqDissipationFittedPowerSpectrum.png')
#     plt.show()
#     freq = [centers[i + 1] - centers[i] for i in np.arange(5)]
#     freqdiff = [x / freq[0] for x in freq]
#     print(freq, freqdiff)
#     print(amplt)
#     ################################################################
#     ################################################################
#     fitting2 = np.polyfit(centers[0:-1:] * delta, freq, 2)
#     plt.scatter(centers, amplt)
#     plt.plot(ell, yy)
#     plt.xlim([0, 2000])
#     plt.ylim([0, None])
#     plt.xlabel('Spherical Harmonic L')
#     plt.ylabel('Intensity (arb. units)')
#     plt.title("Dissipation on High-Fequency CMB Power Spectrum \n delta ={}  Gamma={}".format('%.4E', '%.4E') % (
#         delta, gamma))
#     plt.legend(['Power Spectrum', 'Fitted Data'])
#     plt.savefig('./img1/FreqFitteditted.png')
#     plt.show()
#     ################################################################
#     ################################################################
#     #    A = pars[0]  # sin amplitude
#     # delta = pars[1]  # delta
#     # gamma = pars[2]  #  gamma exponential damping term
#     # h0 = pars[3]
#     parguess = [1.75703174e+00,
#                 -3.18905547e-01, 1.08942482e-02, -1.51470042e-05, 6.98842799e-09,
#                 -7.17548356e-01, 5.52935906e-01, 2.00000000e+01]
#     popt, _ = curve_fit(sindelta, ell[0:2000], dll_SMICA_Clean[0:2000], parguess)
#     print(popt)
#     parguess = np.array(popt)
#     plt.figure()
#     plt.plot(ell, dll_SMICA_Clean)
#     plt.plot(ell, sindelta(ell, *popt), 'r-')
#     plt.xlim([0, 2000])
#     plt.ylim([0, None])
#     plt.xlabel('Spherical Harmonic L')
#     plt.ylabel('Intensity (arb. units)')
#     plt.title("Modeling High-Fequency CMB Power Spectrum")
#     plt.legend(['Power Spectrum', 'Fitted Data'])
#     plt.savefig('./img1/HighFreqFittedPowerSpectrum.png')
#     plt.show()
#     ################################################################
#     ################################################################
#     aaaa = 1


import matplotlib.pylab as plt
import healpy as hp
import numpy as np
import math
from scipy.stats import norm
from scipy.optimize import minimize

thishome = "./DataSupernovaLBLgov/"
# https://irsa.ipac.caltech.edu/data/Planck/release_3/all-sky-maps/matrix_bpasscorr.html
# https://irsa.ipac.caltech.edu/data/Planck/release_2/all-sky-maps/maps/component-maps/cmb/COM_CMB_IQU-smica_1024_R2.02_full.fits

smicafiles = "COM_CMB_IQU-smica_1024_R2.02_full.fits"

beam_arc_min=10
mygauss = fitClass()
planck_IQU_SMICA = hp.fitsfunc.read_map(thishome + smicafiles, dtype= float)
mu_smica, sigma_smica = norm.fit(planck_IQU_SMICA)
nside=1024
white_noise = np.ma.asarray(np.random.normal(0, sigma_smica, 12*nside3D** 2))
cl_WHITE_NOISE, dl_WHITE_NOISE, ell = get_dl(white_noise, nside=nside3D, beam_arc_min=1)
cl_SMICA, dl_SMICA, ell = get_dl(diffmap_lg.squeeze(), nside=nside3D, beam_arc_min=beam_arc_min)

def yy_func(x, ell):
    return x[0] + x[1] * np.exp(ell * x[2] + ell**2 *x[3])

def olderr(x, dl_SMICA, ell):
    err = dl_SMICA - yy_func(x,ell)
    err = np.sum(err * err)
    print(err, x)
    return err
#
#
# mygauss.smica=dll_SMICA[ell>2000]
# mygauss.t = ell[ell>2000]
# mygauss.white_noise = dll_WHITE_NOISE[ell > 2000]
# x0 = np.array([ 0,  1.02461539e-2, -1.00693933e-03])
# mygauss.plotme(x0)
# xout, err = mygauss.optimizeme(x0)
# print(xout, err)
# mygauss.smica=dll_SMICA
# mygauss.t = ell
# mygauss.white_noise = dll_WHITE_NOISE
# adjWN = mygauss.correctWN(xout)
# ymax = np.max(dll_SMICA)
# plt.plot(ell, dll_SMICA, ell, adjWN)
# plt.legend([ "SMICA", "White_Noise"])
# plt.ylim([0,ymax])
# plt.xlim([0,None])
# plt.title("f" + "\n"+ "beam_arc_min={}".format(beam_arc_min))
# plt.show()
# plt.plot(ell, dll_SMICA-adjWN)
# plt.legend([ "Noise-Free SMICA"])
# plt.ylim([0,0.6e-8])
# plt.xlim([0,2000])
# plt.title('f' + "\n"+ "beam_arc_min={}".format(beam_arc_min))
# plt.show()
#
#
#
# # Laplace transform (t->s)
# y = (dll_SMICA-adjWN)[1:2048]
# dl_SMICA_Clean = dll_SMICA-adjWN
# t = ell[1:2048]
# y[y <= 0] = 0.0
# # Frequency domain representation
# samplingFrequency=2048
# # Create subplot
# figure, axis = plt.subplots(2, 1)
# plt.subplots_adjust(hspace=1)
#
# # Time domain representation for sine wave 1
# axis[0].set_title('Original CMB-HF Power Spectrum')
# axis[0].plot(t, y)
# axis[0].set_xlim([0,2048])
# axis[0].set_ylim([0, None])
# axis[0].set_xlabel('L')
# axis[0].set_ylabel('DL')
# # plt.show()
# # Time domain representation for sine wave 2
# fourierTransform = np.fft.fft(y)   # Normalize amplitude
# fourierTransform = fourierTransform[range(int(len(y) / 2))]  # Exclude sampling frequency
# tpCount = len(y)
# values = np.arange(int(tpCount / 2))
# timePeriod = tpCount / samplingFrequency
# frequencies = np.fft.fftshift(values / timePeriod)
# periods = 1/frequencies
# # Frequency domain representation
# axis[1].set_title('Fourier transform depicting the frequency components')
# axis[1].plot(frequencies, abs(fourierTransform))
# axis[1].set_xlabel('Frequencies')
# axis[1].set_ylabel('Amplitude')
# axis[1].set_xlim(0,None)
# axis[1].set_ylim(0, 1e-7)
# plt.show()
#
#
# ax=plt.gca()
# xlim=3000
# ax.set_xlim(0,xlim)
# ymin = np.min(dl_SMICA_Clean[50:xlim])
# ymax = np.max(dl_SMICA_Clean[50:xlim])
# dl_SMICA2= (dl_SMICA_Clean-ymin)/(ymax-ymin)
# yy1= (yy-ymin)/(ymax-ymin)
# # plt.plot(ell, dl_SMICA2, color="green")
# plt.plot(ell, dl_SMICA2, color="green")
# plt.plot(ell, yy, color="red")
# # ax.set_ylim(0,1)
# ax.set_ylim(0,1)
# ax.set_xlim(0,xlim)
# plt.show()
#
#
# if Color.OPTIMIZE_SPECTRUM in todo:
#     (lk, ll, lm) = x0
#     # Create karray
#     karray = list(np.arange(20, 222, 20))
#     karray = np.array(sorted(list(set(karray)))).astype(int)
#     print(len(karray))
#     kmax = max(karray)
#     nside3D = 128
#     bandwidth = 1024
#     myHyper = None
#     myHyper = HYPER(nside3D, sigma_smica, planck_IQU_SMICA, karray,
#                     lk, ll, lm, loadpriorG=True, savePG=True, bandwidth=bandwidth, longG=True)
#     myHyper.SMICA_LR = np.load("./img1/diffmap.npy").T
#     myHyper.plotNewMap(myHyper.SMICA_LR, 0, filename=None, title="DiffMap", plotme=True, save=False, nosigma=False)
#     results, newmap, err, packedresults = myHyper.project4D3d_0(karray)
#     # #################################
#     myHyper.plot_Single_CL_From_Image(newmap, nside3D, xmax=50,log=True)
#     myHyper.plot_Single_CL_From_Image(diffmap, nside3D, xmax=300)
#     myHyper.plot_CL_From_Image(newmap, nside3D, planck_theory_cl, xmax=300, ymax=0)
#
#     newnewmap = myHyper.SMICA_LR.squeeze() - xx0[1] * newmap.squeeze() + xx0[0]
#
#     myHyper.plot_CL_From_Image(diffmap, nside3D, planck_theory_cl, xmax=300, ymax=0.1)
#
#     myHyper.plot_CL_From_Image(myHyper.SMICA_LR.squeeze(), nside3D, planck_theory_cl, xmax=3000, ymax=1)
#     df = pd.DataFrame(np.concatenate([results, myHyper.df[:, 0:4]], axis=1),
#                       columns=["coeff", "k", "l", "m", "std"])
#     filename = "./PG_data/spectrumOptimum_{}_{}__{}_{}_{}".format(kmax, nside3D, chg2ang(lk), chg2ang(ll),
#                                                                   chg2ang(lm))
#     df.to_pickle(filename)
#
#     # subtract from SMICA the background
#     SMICA_LR_NO_BG = myHyper.SMICA_LR.squeeze() - diffmap.squeeze()
#     myHyper.plot_ONLY_CL_From_Image(newmap, nside3D, SMICA_LR_NO_BG, nsidesmica=nside3D, xmax=200)
#
#     df["CL"] = df["coeff"] ** 2
#     df["abscoeff"] = df.coeff.abs()
#
#     fcolors = df.coeff.values
#     (mu, sigma) = norm.fit(fcolors)
#     n, bins, patch = plt.hist(fcolors, 600, density=1, facecolor="r", alpha=0.25)
#     y = norm.pdf(bins, mu, sigma)
#     plt.plot(bins, y)
#     plt.xlim(mu - 4 * sigma, mu + 4 * sigma)
#     plt.xlabel("Amplitude")
#     plt.ylabel("Frequency")
#     plt.ylim(0, 0.04)
#     plt.title("Amplitude Histogram \n HU Modeling of Planck SMICA Map")
#     plt.savefig("./PG_data/AmplitudeHistogram_{}_{}.png", dpi=300)
#     plt.show()
#
#     numberofmodes = df.groupby(['k'])["abscoeff"].count()
#     meanAmplitude = df.groupby(['k'])["abscoeff"].mean()
#     meanEnergy = meanAmplitude ** 2 * meanAmplitude.index * numberofmodes
#     stdd = df.groupby(['k'])["abscoeff"].std()
#     one = np.ones(stdd.shape)
#     valueOfK = stdd.index.values
#
#     # calculation of Predicted Amplitudes
#     predictedAmp = 75 / np.sqrt(valueOfK)
#     # low pass filtering
#     predictedAmp = np.exp(-0.01 * (valueOfK - 2)) * predictedAmp
#     # Frequency dependent freezing phase accounting
#     delta = 0.01
#     predictedAmp = sinxx(valueOfK * delta) * predictedAmp
#     plt.plot(numberofmodes, predictedAmp)
#     plt.scatter(numberofmodes, meanAmplitude)
#     plt.title("Predicted and Observed Mean Amplitudes \n under Energy Equipartition per mode k")
#     plt.xlabel("Accessible Modes for given k")
#     plt.ylabel("Hyperspherical Mode Amplitude per k")
#     plt.ylim(0, 25)
#     plt.xlim(0, 250)
#     plt.savefig("./PG_data/PredicedtAmp.png", dpi=300)
#     plt.show()
#
#     plt.plot(numberofmodes, meanEnergy)
#     plt.title("Mean Energy per k")
#     plt.xlabel("Accessible Modes for given k")
#     plt.ylabel("Mean Energy per k \n arbitrary units")
#     plt.xlim(0, 250)
#     plt.ylim(0, 4.5E5)
#     plt.savefig("./PG_data/MeanEnergy.png", dpi=300)
#     plt.show()
#
#     plt.scatter(numberofmodes, meanAmplitude)
#     plt.title("Mean Amplitude per k")
#     plt.xlabel("Accessible Modes for given k")
#     plt.ylabel("Mean Amplitude per k")
#     plt.ylim(0, 22.5)
#     plt.xlim(0, 250)
#     plt.savefig("./PG_data/MeanAmp.png", dpi=300)
#     plt.show()
#
#     plt.scatter(numberofmodes, stdd)
#     plt.title("STD")
#     plt.xlabel("Accessible Modes for given k")
#     plt.ylabel("STD")
#     plt.xlim(0, 250)
#     plt.savefig("./PG_data/STD.png", dpi=300)
#     plt.show()
#
#     plt.plot(numberofmodes, meanAmplitude / stdd)
#     plt.plot(numberofmodes, np.mean((meanAmplitude / stdd)[3:]) * one)
#     plt.title("MeanAmplitude/std")
#     plt.xlabel("Accessible Modes for given k")
#     plt.ylabel("MeanAmplitude/std")
#     plt.xlim(0, 250)
#     plt.ylim(0, 2)
#     plt.savefig("./PG_data/PredicedtAmpPerSTD.png", dpi=300)
#     plt.show()
#     ################################################################
#     ################################################################
#     ################################################################
#     ################################################################
#     ################################################################
#     ################################################################
#     #################################################################
#
#
# if Color.CREATE_GAUSSIAN_BACKGROUND in todo:
#     nside3D = 128  # 128
#     bandwidth = 3000
#     filename = "./img1/3DGaussianNoise_{}_{}.png".format(kmax, nside3D)
#     title = "3DGaussianNoise_{}_{}".format(kmax, nside3D)
#     (lambda_k, lambda_l, lambda_m) = x0
#     # Create karray
#     karray = list(np.arange(2, 48))
#     # kmax=1
#     # karray += sorted(list(set([int(k) for k in np.geomspace(11,kmax,30)])))
#     print(len(karray))
#     kmax = max(karray)
#     #################################
#     mypath = "./PG_data"
#     myHyper = HYPER(nside3D, sigma_smica, planck_IQU_SMICA, karray, mypath,
#                     lambda_k, lambda_l, lambda_m, loadpriorG=True, savePG=True)
#     #################################################################
#     delta = 0.001
#     lmax = 1000
#     llist = sorted(list(set([int(k) for k in np.geomspace(1, lmax + 1, 100)])))
#     wavefunction = myHyper.createGaussianBackground(x0, nside3D, delta, karray=llist)
#     wavefunction = myHyper.normalizeFColors(wavefunction, myHyper.sigma_smica).squeeze()
#     myHyper.plotNewMap(wavefunction, 0, filename=filename, title=title, plotme=True, save=True)
#     myHyper.plot_CL_From_Image(wavefunction, nside3D, planck_theory_cl, xmax=2 * lmax, ymax=0.01)
#     myHyper.plot_ONLY_CL_From_Image(wavefunction, nside3D, myHyper.SMICA, nsidesmica=1024, xmax=2 * lmax)
#     np.save("./PG_data/3DGaussianNoise_{}_{}.npy".format(kmax, nside3D), wavefunction, allow_pickle=True)
# ###############################################################
#
#
# if Color.FINDBESTFORKRANGE in todo:
#     # myHyper.phi, myHyper.theta, myHyper.ksi = hp.pix2vec(nside=nside3D, ipix=np.arange(hp.nside2npix(nside=nside3D)))
#     # myHyper.theta, myHyper.phi = hp.pix2ang(nside=nside3D,
#     #                                                      ipix=np.arange(hp.nside2npix(nside=nside3D)))
#     # myHyper.costheta = np.cos(myHyper.theta)
#     # k = 9
#     # l = 3
#     # for m in np.arange(-l+1,l):
#     #     pp = np.array([legendre(l, x) for x in myHyper.costheta])
#     #     myHyper.plot_aitoff_df(l,m,myHyper.phi, pp=pp)
#     #
#     # x0=np.load("./img1/x0_{}_{}.npy".format(kmax, nside3D), allow_pickle=True)
#     # x00 = np.load("./img1/x00_{}_{}.npy".format(kmax, nside3D), allow_pickle=True)
#     # x0 = np.load("./img1/results_{}_{}.npy".format(kmax, nside3D), allow_pickle=True)
#     # x0 = [5.45066325E+00, 1.36266581E+00, -2.65625000E-04]
#     (lk, ll, lm) = x0
#     myHyper.change_SMICA_resolution(nside3D, myHyper.sigma_smica, bandwidth=bandwidth)
#     olderr = 1110.0
#
#     for kk in [x for x in karray if x > 10]:
#         print(kk)
#         kkarray = [x for x in karray if x <= kk + 1]
#         kmax = np.max(kkarray)
#         results, newmap, err, packedresults = myHyper.project4D3d_0(kkarray)
#         print(lk, ll, lm, err, "intermediate result")
#         if err < olderr:
#             olderr = err
#             filename = "./img1/Best_{}_{}__{}_{}_{}_{}.png".format(kmax, nside3D, np.round(lk, 1),
#                                                                    np.round(ll, 1), np.round(lm, 1),
#                                                                    np.round(err * 1, 4))
#             title = "Best_{}_{}__{}_{}_{}_{}".format(kmax, nside3D, np.round(lk, 1),
#                                                      np.round(ll, 1), np.round(lm, 1),
#                                                      np.round(err * 1, 4))
#             myHyper.plotNewMap(newmap, err, filename=filename, title=title, plotme=True, save=True)
#             print(lk, ll, lm, err, title)
#             filenameResults = "./PG_data/Results_{}_{}__{}_{}_{}_{}.npy".format(kmax, nside3D, np.round(lk, 1),
#                                                                                 np.round(ll, 1),
#                                                                                 np.round(lm, 1),
#                                                                                 np.round(err * 1, 4))
#             np.save(filenameResults, packedresults)
#             np.save("./PG_data/resultsBest_{}_{}.npy".format(kmax, nside3D), results, allow_pickle=True)
#             myHyper.plotHistogram(newmap.squeeze(), nside3D, kmax, plotme=True)
#             myHyper.plotNewMap(newmap, err, filename=filename, title=title, plotme=True, save=False)
#     # dl_SMICA, dl_SMICA_HU, ell = myHyper.plot_ONLY_CL_From_Image(newmap.squeeze(), bandwidth, xmax=10 * kmax)
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import cmath
# def A(l,L,alpha, beta):
#     i = 1.J
#     l=l*np.pi
#     a1 = 1/np.sqrt(2*np.pi)/((np.log(2)/L)-i*l)*(np.exp(((np.log(2)/L)-i*l)*L)-1)
#     a2 = np.sqrt(2/pi)*(2/L)*np.exp(-i*l*L/2)*np.sin(l*L/2)/(l*L/2)
#     return np.abs(a1-a2)**2*l**alpha*np.exp(-beta*l)
#
# l = np.arange(2000)
# L=1/200
# beta=1/1600
# alpha= 2.3 #2000*beta
# power = A(l,L,alpha, beta)
# plt.plot(l,power)
# plt.show()
#
# beam_arc_min=10
# white_noise = np.ma.asarray(np.random.normal(0, sigma_smica, 12*nside3D** 2))
# cl_WHITE_NOISE, dl_WHITE_NOISE, ell = get_dl(white_noise, nside=nside3D, beam_arc_min=1)
# cl_SMICA, dl_SMICA, ell = get_dl(diffmap_lg.squeeze(), nside=nside3D, beam_arc_min=beam_arc_min)
# mygauss.smica=dl_SMICA[ell>2000]
# mygauss.t = ell[ell>2000]
# mygauss.white_noise = dl_WHITE_NOISE[ell > 2000]
# x0 = np.array([ 0,  1.02461539e-2, -1.00693933e-03])
# mygauss.plotme(x0)
# xout, err = mygauss.optimizeme(x0)
# print(xout, err)
# mygauss.smica=dl_SMICA
# mygauss.t = ell
# mygauss.white_noise = dl_WHITE_NOISE
# adjWN = mygauss.correctWN(xout)
# ymax = np.max(dl_SMICA)
# dl_WHITE_NOISE_Clean = dl_SMICA-adjWN
# plt.plot(ell, dl_SMICA, ell, adjWN)
# plt.legend([ "SMICA", "White_Noise"])
# plt.ylim([0,ymax])
# plt.xlim([0,None])
# plt.title("f" + "\n"+ "beam_arc_min={}".format(beam_arc_min))
# plt.show()
# plt.plot(ell, dl_WHITE_NOISE_Clean)
# plt.legend([ "Noise-Free SMICA"])
# plt.ylim([0,0.6e-8])
# plt.xlim([0,2000])
# plt.title("f" + "\n"+ "beam_arc_min={}".format(beam_arc_min))
# plt.show()
#
#
#
#
# # cl_SMICA, dl_SMICA, ell, xx0 = myHyper.plot_CL_From_Image(fcolors=dl_WHITE_NOISE_Clean.squeeze(), nside=nside3D, planck_theory_cl=planck_theory_cl, xmax=300, xlim=2048, log=False)
#
#
#
#
#
#
