import sys

def testingcode():
	from numpy import load,shape

	mat = "25000pca.mat"
	xtrans=load(mat)
	mu=simple_mu(xtrans)
	Wred = run_pca(xtrans,mu,n_comp=10)
	Wred.dump('pca.mat')

def testingcode_1(saved="",mat=""):
	if not saved=="":
		from numpy import load
		W_red = load(saved)
	elif not mat=="":
		load_timeseries(mat=mat)
	else:
		waves, xtrans = load_timeseries(name="all")
		mu = simple_mu(xtrans)
		W_red = run_pca(xtrans,mu,n_comp=29)
		W_red.dump('pca.mat')
	for i in range(20):
		name=str(i)+".png"
		pca_spec(xtrans[i],W_red,mu,plot=True,plotname=name)


def simple_mu(xtrans):
	from numpy import mean,shape
	mu = mean(xtrans,axis=0)
	mu = mu/(mu.dot(mu))
	return mu

def pca_spec(spec,W_red,mu,plot=False,plotname="spec.png"):
	spec = spec/(spec.dot(spec))
	a = (W_red.T).dot(spec.T)
	model = W_red.dot(a)+mu
	if plot:
		from pylab import clf,plot,show,savefig
		waves=range(len(model))
		plot(waves,model,'-r',waves,spec,'.')
		savefig(plotname)
		clf()


def run_pca(xtrans,mu,n_comp=10):
	print("Running PCA with ",n_comp, " components")
	from numpy import linalg,around, diag, zeros,shape
	[n,m] = shape(xtrans)
	X = (demean_pca(xtrans,mu)).T
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

def load_timeseries(name="DD2D_asym_01_dc2",mat=""):

	if not mat=="":
		lumis = load_mat(mat)
		fcut = int(mat.strip("pca.mat"))
		fsearch="data/DD2D*/*/*/*"
		print("starting data collection at ",fcut)
	elif name=="all":
		fsearch="data/DD2D*/*/*/*"
		fcut=0
	else:
		fsearch="data/"+name+"/*/*/*"
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
		[wave,lum] = genfromtxt(f,usecols=(0,1),unpack=True)
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
			name = str(ctr)+"pca.mat"
			lumis.dump(name)
		
		wavecheck = wave
	lumis.dump('lumis')
	return waves, lumis
		
def demean_pca(timeseries,mu):
	from numpy import mean,array,dot,shape
	times = [a/(a.dot(a.T)) for a in timeseries]
	ts = array([Xvec - mu for Xvec in times])
	return ts





