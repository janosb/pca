def testingcode():
        load_timeseries()


def load_timeseries(name="DD2D_asym_01_dc2"):
	if name=="all":
		fsearch="data/DD2D*/*/*/*"
	else:
		fsearch="data/"+name+"/*/*/*"

	#find files
	import glob
	from numpy import genfromtxt
	fnames=glob.glob(fsearch)

	#build the output array
	from numpy import array,append
	
	lumis = array([])
	waves = array([])
	wavecheck=array([1000])
	for f in fnames:
		[wave,lum] = genfromtxt(f,usecols=(0,1),unpack=True)
		print(f,wave,waves)
		if not all(wave == wavecheck):
			print("New wavelength array detected")
			waves.append(wave)
		lumis.append(lum)
		wavecheck = wave
		
		



def demean_pca(timeseries,mu):
	from numpy import mean,array,dot,shape
	times = [a/(a.dot(a.T)) for a in timeseries]
	ts = array([Xvec - mu for Xvec in times])
	return ts





