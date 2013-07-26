import sys

def testingfits():
	import pylab as pl
	import numpy as np
	xtrans = load_timeseries_fits(all=True)
	mu = simple_mu(xtrans)
	print np.shape(mu)


def testingsvd():
	from numpy import load,shape,linalg

	mat = "/global/homes/j/janosb/odetta/data/20000pca.mat"
	xtrans=load(mat)
	mu=simple_mu(xtrans)
	X = (demean_pca(xtrans,mu)).T
	[W, s, V_T] = linalg.svd(X)
	W.dump("/global/homes/j/janosb/odetta/W.mat")
	s.dump("/global/homes/j/janosb/odetta/s.mat")
	V_T.dump("/global/homes/j/janosb/odetta/V_T.mat")

def testingcode():
	from numpy import load,shape,zeros,mean
	lumin = "/global/homes/j/janosb/odetta/data/20000pca.mat"
	mat = "/global/homes/j/janosb/odetta/W_20000.mat"
 	n_comp=10
	W = load(mat)
	W_red = zeros(shape(W))
	for i in range(n_comp):
 	     	W_red[:,i] = W[:,i]
	xtrans=load(lumin)
	#mu = simple_mu(xtrans)
	mu = load('/global/homes/j/janosb/odetta/data/20000mu.mat')
	for i in range(100):
		name="/global/homes/j/janosb/odetta/plots/"+str(i)+".png"
		pca_spec(xtrans[i],W_red,mu,plot=True,plotname=name)

def testingcode_1(saved="",mat=""):
	if not saved=="":
		from numpy import load
		W_red = load(saved)
	elif not mat=="":
		load_timeseries(mat=mat)
	else:
		waves, xtrans = load_timeseries(name="all")

def simple_mu(xtrans):
	from numpy import mean,shape
	'''Here we expect xtrans to be a 3D array'''
	mu = mean(mean(xtrans,axis=0),axis=1)
	mu = mu/(mu.dot(mu))
	return mu

def pca_spec(spec,W_red,mu,plot=False,plotname="spec.png"):
	spec = spec/(spec.dot(spec))-mu
	a = (W_red.T).dot(spec.T)
	model = W_red.dot(a)+mu
	spec+=mu
	if plot:
		from pylab import clf,plot,ylabel,subplot2grid,savefig
		waves=range(len(model))
		subplot2grid((3,3),(0,0),colspan=3,rowspan=2)
		ylabel('Scaled Flux')
		plot(waves,spec,'.',waves,model,'-r')
		subplot2grid((3,3),(2,0),colspan=3)
		plot(waves, (model-spec)/spec,".")
		ylabel('Residual (%)')
		savefig(plotname)
		clf()


def run_pca(xtrans,mu,n_comp=10):
	print("Running PCA with ",n_comp, " components")
	from numpy import linalg,around, diag, zeros,shape,array
	import numpy as np
	xtrans=array(xtrans,dtype=np.float32)
	[n,m] = shape(xtrans)
	X = (demean_pca(xtrans,mu)).T
	X = array(X, dtype=np.float32)
	[W, s, V_T] = linalg.svd(X)
	s = diag(s)
	S = zeros([m,n])
	S[:n,:n] = s
	#X_pca = W.dot(S).dot(V_T) # should be 0 vector
	W_reduced = zeros(shape(W))
	for i in range(n_comp):
		W_reduced[:,i] = W[:,i]

	return W_reduced

def load_mat(name):
	from numpy import load
	lumis = load(name)
	return lumis

def load_timeseries_fits(name="DD2D_asym_01_dc2_mu014.fits",all=False):
	import pyfits as pf
	import pylab as pl
	from numpy import float,zeros,shape,append
	datadir = "/Users/janos/Desktop/grad/pca/fits/fits_v001/"
	if all:
		import glob
		fnames = glob.glob(datadir+"*")
		data=zeros((51,551,len(fnames)),dtype=float)
		for ix,f in enumerate(fnames):
			hud = pf.open(f)
			data[:,:,ix]=hud[0].data
		return data
	else:
		hdu = pf.open(name)
		return hdu[0].data
	

def load_timeseries(name="DD2D_asym_01_dc2",mat=""):

	if not mat=="":
		lumis = load_mat('/global/homes/j/janosb/odetta/'+mat)
		fcut = int(mat.strip("pca.mat"))
		fsearch="/global/homes/j/janosb/odetta/data/DD2D*/*/*/*"
		print("starting data collection at ",fcut)
	elif name=="all":
		fsearch="/global/homes/j/janosb/odetta/data/DD2D*/*/*/*"
		fcut=0
	else:
		fsearch="/global/homes/j/janosb/odetta/data/"+name+"/*/*/*"
		fcut=0

	#find files
	import glob
	fnames=glob.glob(fsearch)
	ntot=len(fnames)
	print('found '+str(ntot)+' spectra to run')
	#build the output array
	from numpy import genfromtxt,max, array,append,vstack,shape
	
	ctr=fcut
	wavecheck=array([0])
	for f in fnames[fcut:]:
		ctr+=1
		try:
			[wave,lum] = genfromtxt(f,usecols=(0,1),unpack=True)
		except ValueError:
			print("failed to read "+f)
			sys.exit()
		try: 
			bb = all(wave==wavecheck)
		except TypeError:
			print("failed at "+f)
			sys.exit()
		if not all(wave == wavecheck):
			print("New wavelength array detected at "+f)
			try:
				waves=vstack((waves,wave))
			except UnboundLocalError:
				waves=wave
			
		try:
			lumis=vstack((lumis,lum))
		except UnboundLocalError:
			lumis=lum

		if (ctr %1000)==0:
			print(ctr,"/",ntot)
		if (ctr %5000)==0:
			name = "/global/homes/j/janosb/odetta/"+str(ctr)+"pca.mat"
			lumis.dump(name)
		
		wavecheck = wave
	lumis.dump('/global/homes/j/janosb/odetta/lumis.mat')
	return waves, lumis
		
def demean_pca(timeseries,mu):
	from numpy import mean,array,dot,shape
	times = [a/(a.dot(a.T)) for a in timeseries]
	ts = array([Xvec - mu for Xvec in times])
	return ts





if __name__=="__main__":
	testingfits()



