import numpy as np
import pylab as pl
import pyfits as pf

def readfile(name):
    wave,flux = np.genfromtxt(name, dtype=np.double, unpack=True)
    return wave, flux

def interpl(wave1,wave2,flux2):
    flux1 = np.interp(wave1,wave2,flux2)
    return flux1

def fitsdata(name):
    hdu = pf.open(name)
    coeff0 = 3.45
    coeff1 = .001
    naxis1 = 551
    wave = [10**(coeff0+coeff1*i) for i in range(naxis1)]
    fluxes = hdu[0].data
    return wave, fluxes

def testdata():
    f1 = "fits/fits_v001/DD2D_asym_06_dc3_mu165.fits"
    f2 = "hsiao_template/spectra/0.dat"
    wave1, fluxes = fitsdata(f1)
    flux1 = fluxes[0]
    wave2, flux2 = np.genfromtxt(f2,unpack=True,usecols=(1,2),dtype=np.double)
    return wave1,flux1,wave2,flux2
    
if __name__=="__main__":
    wave1,flux1,wave2,flux2 = testdata()
    flux = interpl(wave1, wave2,flux2)

    pl.clf()
    pl.plot(wave2,flux2, '-',wave1,flux,'.')
    pl.show()

