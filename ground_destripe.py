import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import time
from tqdm import trange, tqdm
import scipy
from scipy.sparse import csr_matrix, csc_matrix
from scipy.sparse.linalg import inv as sparse_inv
import matplotlib
import h5py
import argparse
# from sparse_dot_mkl import dot_product_mkl, gram_matrix_mkl, sparse_qr_solve_mkl
matplotlib.rcParams.update({'font.size': 16})
matplotlib.rcParams['figure.figsize'] = (10.0, 5.0)
import warnings
warnings.filterwarnings("ignore")

# WN_INTERVAL = 1000
TOL = 1e-10
WN_INTERVAL = "PER_SCAN"
OBSID_RANGE = [0, 20000]
# OBSID_RANGE = [0,10540]
# OBSID_RANGE = [10541, 20000]


class TODhandler:
    def __init__(self, el_range, az_range, Nfeeds, sb):
        self.tsys_database = h5py.File("/mn/stornext/d16/cmbco/comap/jonas/comap_general/tsys/level1_database_old.h5", "r")
        self.el_range = el_range
        self.az_range = az_range
        self.Nfeeds = Nfeeds
        self.sb = sb

    def load_data_from_files(self, filenames):
        Nmax = int(1e10)
        Nfiles = len(filenames)
        tod = np.zeros(Nmax)
        el = np.zeros(Nmax)
        az = np.zeros(Nmax)
        obsid_idxs = []
        scan_idxs = []
        obsids = []
        nr_success = 0
        tod_idx = 0
        for feed in range(self.Nfeeds):
            for i, filename in tqdm(enumerate(filenames)):
                _tod, _el, _az, _scan_idxs, obsid = self.load_data_from_file(filename, feed=feed)
                _Ntod = _tod.shape[-1]
                tod[tod_idx:tod_idx+_Ntod] = _tod
                el[tod_idx:tod_idx+_Ntod] = _el
                az[tod_idx:tod_idx+_Ntod] = _az
                if _Ntod > 0:
                    #tod[tod_idx:tod_idx+_Ntod] -= np.mean(tod[tod_idx:tod_idx+_Ntod])
                    scan_idxs.extend(_scan_idxs + tod_idx)  # add tod_idx to go from "local idx" to "global idx".
                    obsid_idxs.append(tod_idx)
                    obsids.append(int(obsid))
                    nr_success += 1
                tod_idx += _Ntod
        print(f"Finished TOD load with {nr_success} successful obsids, and total sample size {tod_idx}.")
        tod = tod[:tod_idx]
        el = el[:tod_idx]
        az = az[:tod_idx]
        self.tod, self.el, self.az, self.scan_idxs, self.obsid_idxs, self.obsids = tod, el, az, np.array(scan_idxs), np.array(obsid_idxs), np.array(obsids, dtype=int)

    def load_data_from_file(self, filename, feed):
        try:
            obsid = filename.split("/")[-1].split("-")[1][1:]
        except:  # Filename does not follow convention.
            return np.array([]), np.array([]), np.array([]), np.array([]), ""
        if not obsid in obsid_dict:  # File is not in runlist, so we don't know the scan star and end points.
            return np.array([]), np.array([]), np.array([]), np.array([]), ""
        if not "0"+obsid in self.tsys_database["obsid"]:
            return np.array([]), np.array([]), np.array([]), np.array([]), ""
        tod = self.tsys_database["obsid/0" + obsid + "/tod_sbmean"][feed, self.sb]
        
        with h5py.File(filename, "r") as f:
            #tod        = f["/spectrometer/band_average"][feed, self.sb]
            tod_times_MJD  = f["/spectrometer/MJD"][()]
            el = f["/spectrometer/pixel_pointing/pixel_el"][feed]
            az = f["/spectrometer/pixel_pointing/pixel_az"][feed]
            #feeds           = f["/spectrometer/feeds"][:18]

        if el.shape != tod.shape: # Dunno why this happens, but it happens sometimes.
            print(f"Error: Size mismatch between databases in file {filename}")
            return np.array([]), np.array([]), np.array([]), np.array([]), ""
        cut_idx = (~(np.isnan(tod)))*(self.el_range[0] < el)*(self.el_range[1] > el)*(self.az_range[0] < az)*(self.az_range[1] > az)
        cut_idx *= (tod > 20)*(tod < 100)
        tod = tod[cut_idx]
        tod_times_MJD = tod_times_MJD[cut_idx]
        el = el[cut_idx]
        az = az[cut_idx]
        if tod_times_MJD.shape[-1] < 1000:  # Too few points.
            return np.array([]), np.array([]), np.array([]), np.array([]), ""

        scan_times = obsid_dict[obsid]
        scan_idxs = np.zeros_like(scan_times, dtype=int)
        for i in range(len(scan_times)):
            scan_idxs[i,0] = np.argmin(np.abs(scan_times[i][0] - tod_times_MJD))
            scan_idxs[i,1] = np.argmin(np.abs(scan_times[i][1] - tod_times_MJD))
        cut_slice = slice(scan_idxs[0,0], scan_idxs[-1,-1])
        tod = tod[cut_slice]
        el = el[cut_slice]
        az = az[cut_slice]
        scan_idxs = np.array(scan_idxs) - scan_idxs[0][0] # Compensate for that we cut away everything before first scan.
                                                               # Also now only holds start of each scan.

        return tod, el, az, scan_idxs, obsid


class Mapmaker:
    def __init__(self):
        self._Corr_wn = None
        self._Corr_wn_inv = None

    def TOD_from_arrays(self, TOD, samprate, Nsidemap=80, norm_scale=None):
        self.y = TOD.tod
        self.el = TOD.el
        self.az = TOD.az
        self.obsid_idxs = TOD.obsid_idxs
        self.Nobsids = len(self.obsid_idxs)
        self.scan_idxs = TOD.scan_idxs
        self.Nscans = len(self.scan_idxs)
        self.samprate = samprate
        #self.y -= np.mean(self.y)
        Ntod = self.y.shape[-1]
        el_min, el_max = self.el.min(), self.el.max()
        az_min, az_max = self.az.min(), self.az.max()
        el_range = el_max - el_min
        az_range = az_max - az_min
        Nmap = Nsidemap**2

        el_step = el_range/(Nsidemap-1)  #TODO: Better binning
        az_step = az_range/(Nsidemap-1)
        y_idx = np.array((self.el-el_min)/el_step, dtype=int)
        x_idx = np.array((self.az-az_min)/az_step, dtype=int)
        self.P = csc_matrix((np.ones(Ntod), (np.arange(Ntod, dtype=int), x_idx+Nsidemap*y_idx)), shape=(Ntod, Nmap))
        self.Ntod, self.Nsidemap, self.Nmap = Ntod, Nsidemap, Nmap
        del(y_idx, x_idx)
    
    def make_basis(self, basis_type="constant", basis_size=1000):
        t0 = time.time();
        Ntod = self.Ntod
        Nbasis = int(np.ceil(Ntod/basis_size))
        self.Nbasis, self.basis_size = Nbasis, basis_size
   
        if basis_type == "constant":
            col_idx = np.arange(Ntod, dtype=int)
            row_idx = col_idx//basis_size
            data = np.ones(Ntod)
        elif basis_type == "constant_plus_el":
            col_idx = np.zeros(2*Ntod, dtype=int)
            row_idx = np.zeros(2*Ntod, dtype=int)
            data = np.zeros(2*Ntod)
            col_idx[0:Ntod] = np.arange(Ntod, dtype=int)
            row_idx[0:Ntod] = np.arange(Ntod, dtype=int)//basis_size
            col_idx[Ntod:2*Ntod] = np.arange(Ntod, dtype=int)
            row_idx[Ntod:2*Ntod] = Nbasis
            data[0:Ntod] = np.ones(Ntod)
            data[Ntod:2*Ntod] = 1.0/np.sin(self.el*np.pi/180.0)
            Nbasis += 1
        elif basis_type == "constant_plus_el_per_obsid":
            col_idx = np.zeros(2*Ntod, dtype=int)
            row_idx = np.zeros(2*Ntod, dtype=int)
            data = np.zeros(2*Ntod)
            col_idx[0:Ntod] = np.arange(Ntod, dtype=int)
            row_idx[0:Ntod] = np.arange(Ntod, dtype=int)//basis_size
            col_idx[Ntod:2*Ntod] = np.arange(Ntod, dtype=int)
            for i in range(self.Nobsids-1):
                row_idx[Ntod+self.obsid_idxs[i]:Ntod+self.obsid_idxs[i+1]] = Nbasis+i
            row_idx[Ntod+self.obsid_idxs[-1]:] = Nbasis+self.Nobsids-1
            data[0:Ntod] = np.ones(Ntod)
            data[Ntod:2*Ntod] = 1.0/np.sin(self.el*np.pi/180.0)
            Nbasis += self.Nobsids
            # Creating Covariance matrix of a, for prior.
            mean = 20
            sigma2 = 0.1
            self.Ca_inv = np.zeros(Nbasis)
            self.Ca_inv_mua = np.zeros(Nbasis)
            self.Ca_inv[-self.Nobsids:] = 1.0/sigma2
            self.Ca_inv = scipy.sparse.diags(self.Ca_inv)
            self.Ca_inv_mua[-self.Nobsids:] = mean/sigma2

        self.F = csc_matrix((data, (col_idx, row_idx)), shape=(Ntod, Nbasis))
        del(col_idx, row_idx, data)
        self.Nbasis, self.basis_size = Nbasis, basis_size
        print(f"Basis creation:   {time.time()-t0:.2f}s"); 

    def map_destripe(self):
        t2 = time.time()
        y, P, F, Corr_wn, Corr_wn_inv, Ntod, Nbasis, basis_size = self.y, self.P, self.F, self.Corr_wn, self.Corr_wn_inv, self.Ntod, self.Nbasis, self.basis_size
        
        t1=time.time()
        t0=time.time()
        PT = csc_matrix(P.T)
        print(f"PT:   {time.time()-t0:.2f}s"); t0 = time.time()
        FT = csc_matrix(F.T)
        print(f"FT:   {time.time()-t0:.2f}s"); t0 = time.time()
        inv_PT_C_P = scipy.sparse.diags( 1.0/(PT.dot(Corr_wn_inv).dot(P)).diagonal() )
        #inv_PT_C_P = scipy.sparse.diags(1.0/(dot_product_mkl(dot_product_mkl(PT, Corr_wn_inv), P)).diagonal())
        print(f"inv_PT_C_P:   {time.time()-t0:.2f}s"); t0 = time.time()
        P_inv_PT_C_P = P.dot(inv_PT_C_P)
        # P_inv_PT_C_P = dot_product_mkl(P, inv_PT_C_P)
        print(f"P_inv_PT_C_P:   {time.time()-t0:.2f}s"); t0 = time.time()
        FT_C_F = FT.dot(Corr_wn_inv).dot(F)
        #FT_C_F = dot_product_mkl(dot_product_mkl(FT, Corr_wn_inv), F)
        print(f"FT_C_F:   {time.time()-t0:.2f}s"); t0 = time.time()
        FT_C_P_inv_PT_C_P = FT.dot(Corr_wn_inv.dot(P_inv_PT_C_P))
        #FT_C_P_inv_PT_C_P = dot_product_mkl(dot_product_mkl(FT, Corr_wn_inv), P_inv_PT_C_P)
        print(f"FT_C_P_inv_PT_C_P:   {time.time()-t0:.2f}s"); t0 = time.time()
        PT_C_F = PT.dot(Corr_wn_inv).dot(F)
        #PT_C_F = dot_product_mkl(dot_product_mkl(PT, Corr_wn_inv), F)
        print(f"PT_C_F:   {time.time()-t0:.2f}s"); t0 = time.time()
        del(FT)
        print(f"Setting up matrices:   {time.time()-t1:.2f}s"); t0=time.time()

        def LHS(a):
            a1 = FT_C_F.dot(a)
            a2 = PT_C_F.dot(a)
            a3 = FT_C_P_inv_PT_C_P.dot(a2)
            a4 = self.Ca_inv.dot(a)
            
            # return a1 - a3 + a4
            return a4
        A = scipy.sparse.linalg.LinearOperator((Nbasis,Nbasis), matvec=LHS)
        # b = F.T.dot(Corr_wn_inv).dot(y) - FT_C_P_inv_PT_C_P.dot(PT.dot(Corr_wn_inv).dot(y)) + self.Ca_inv_mua
        b = self.Ca_inv_mua
        print(f"Setting up A and b:   {time.time()-t0:.2f}s"); t0=time.time()
        
        precond_array = np.zeros((Nbasis,))
        # precond_array += FT_C_F.diagonal()
        precond_array += self.Ca_inv.diagonal()
        FT_C_P_inv_PT_C_P_rowformat = csr_matrix(FT_C_P_inv_PT_C_P)
        # for i in range(Nbasis):
            # asdf =  PT_C_F.getcol(i).toarray().reshape(PT_C_F.shape[0]).dot(FT_C_P_inv_PT_C_P.getrow(i).toarray().reshape(FT_C_P_inv_PT_C_P.shape[-1]))
            # precond_array[i] += FT_C_P_inv_PT_C_P_rowformat.getrow(i).dot(PT_C_F.getcol(i)).data[0]
        M = scipy.sparse.diags(1.0/precond_array)
        print(f"Setting up preconditioner:   {time.time()-t0:.2f}s"); t0=time.time()
        
        def solve_cg(A, b):
            num_iter = 0
            def callback(xk):
                nonlocal num_iter
                if num_iter%100 == 0:
                    print(num_iter)
                    np.save(f"data/a_{num_iter//100}.npy", xk)
                num_iter += 1
            x0 = np.zeros(Nbasis)
            x0[-self.Nobsids:] = 12.0
            # a, info = scipy.sparse.linalg.cg(A, b, callback=callback, tol=TOL, M=M, x0=x0)
            a, info = scipy.sparse.linalg.cg(A, b, callback=callback, tol=TOL, x0=x0)
            return a, info, num_iter
        self.a, info, num_iter = solve_cg(A, b)
        print(f"Num iterations in CG:   {num_iter}")
        print(f"CG info:   {info}")
        print(f"Solve a with CG:   {time.time()-t0:.2f}s"); t0=time.time()
        self.map_nw = inv_PT_C_P.dot(PT.dot(Corr_wn_inv).dot(y))
        self.map_destripe = inv_PT_C_P.dot(PT.dot(Corr_wn_inv).dot(y-F.dot(self.a)))
        self.template_map = inv_PT_C_P.dot(PT.dot(Corr_wn_inv).dot(F.dot(self.a)))
        self.el_template_map = inv_PT_C_P.dot(PT.dot(Corr_wn_inv).dot(F[:,-self.Nobsids:]*self.a[-self.Nobsids:])) # Only elevation template.
        self.hitmap = PT.dot(P).diagonal()
        print(f"map calc:   {time.time()-t0:.2f}s")
        print(f"map destripe:   {time.time()-t2:.2f}s")
        del(A, b, PT, inv_PT_C_P, P_inv_PT_C_P, FT_C_F, FT_C_P_inv_PT_C_P, PT_C_F)

    @property
    def Corr_wn(self):
        y, Ntod, Nscans, scan_idxs, samprate = self.y, self.Ntod, self.Nscans, self.scan_idxs, self.samprate
        if self._Corr_wn is None:
            if WN_INTERVAL == "PER_SCAN":
                Corr_wn = np.zeros(Ntod)
                Corr_wn_small = np.zeros(Nscans)
                for i in range(Nscans-1):
                    y_scan = y[scan_idxs[i,0]:scan_idxs[i+1,0]]
                    Corr_wn_small[i] = np.var(y_scan[1:] - y_scan[:-1])/np.sqrt(samprate)
                    Corr_wn[scan_idxs[i,0]:scan_idxs[i+1,0]] = Corr_wn_small[i]
                y_scan = y[scan_idxs[i+1,0]:]
                Corr_wn_small[i+1] = np.var(y_scan[1:] - y_scan[:-1])/np.sqrt(samprate)
                Corr_wn[scan_idxs[i+1,0]:] = Corr_wn_small[i+1]
            else:
                Corr_wn_small = np.zeros(Ntod//WN_INTERVAL)
                Corr_wn = np.zeros(Ntod)
                for i in range(Ntod//WN_INTERVAL - 1):
                    y_scan = y[i*WN_INTERVAL:(i+1)*WN_INTERVAL]
                    Corr_wn_small[i] = np.var(y_scan[1:] - y_scan[:-1])/np.sqrt(samprate)
                    Corr_wn[i*WN_INTERVAL : (i+1)*WN_INTERVAL] = Corr_wn_small[i]
                y_scan = y[(i+1)*WN_INTERVAL:]
                Corr_wn_small[i+1] = np.var(y_scan[1:] - y_scan[:-1])/np.sqrt(samprate)
                Corr_wn[(i+1)*WN_INTERVAL:] = Corr_wn_small[i]

            np.save(f"Corr_wn_{WN_INTERVAL}.npy", Corr_wn)
            np.save(f"Corr_wn_small_{WN_INTERVAL}.npy", Corr_wn_small)
            self._Corr_wn = scipy.sparse.diags(Corr_wn, format="csc")
            self._Corr_wn_inv = scipy.sparse.diags(1.0/np.where(Corr_wn>0, Corr_wn, 1), format="csc")
            self.Corr_wn_small = Corr_wn_small
            return self._Corr_wn
        else:
            return self._Corr_wn

    @property
    def Corr_wn_inv(self):
        if self._Corr_wn_inv is None:
            self.Corr_wn
            return self._Corr_wn_inv
        else:
            return self._Corr_wn_inv


def make_plots(filename):
    with h5py.File("data/" + filename + ".hd5", "r") as f:
        el_range = f["el_range"][()]
        az_range = f["az_range"][()]
        map_destripe = f["map_destripe"][()]
        map_hits = f["map_hits"][()]
        map_nw = f["map_nw"][()]
        map_full_template = f["map_full_template"][()]
        map_el_template = f["map_el_template"][()]

    threshold = map_hits.max()/100
    map_destripe[map_hits < threshold] = np.nan
    map_nw[map_hits < threshold] = np.nan
    map_full_template[map_hits < threshold] = np.nan
    map_el_template[map_hits < threshold] = np.nan
    map_destripe_template = map_full_template-map_el_template
    
    import copy
    cmap = copy.copy(plt.get_cmap("viridis"))
    cmap.set_bad(color = "grey", alpha = 1.)

    fig, ax = plt.subplots(5, 1, figsize=(18,24))
    img0 = ax[0].pcolormesh(az_range, el_range, map_nw, cmap=cmap, vmin=np.nanmean(map_nw) - 2*np.nanstd(map_nw), vmax=np.nanmean(map_nw) + 2*np.nanstd(map_nw))
    img1 = ax[1].pcolormesh(az_range, el_range, map_destripe, cmap=cmap, vmin=np.nanmean(map_destripe) - 2*np.nanstd(map_destripe), vmax=np.nanmean(map_destripe) + 2*np.nanstd(map_destripe))
    img2 = ax[2].pcolormesh(az_range, el_range, map_destripe_template, cmap=cmap, vmin=np.nanmean(map_destripe_template) - 2*np.nanstd(map_destripe_template), vmax=np.nanmean(map_destripe_template) + 2*np.nanstd(map_destripe_template))
    img3 = ax[3].pcolormesh(az_range, el_range, map_el_template, cmap=cmap, vmin=np.nanmean(map_el_template) - 2*np.nanstd(map_el_template), vmax=np.nanmean(map_el_template) + 2*np.nanstd(map_el_template))
    img4 = ax[4].pcolormesh(az_range, el_range, map_hits)
    plt.colorbar(img0, ax=ax[0])
    plt.colorbar(img1, ax=ax[1])
    plt.colorbar(img2, ax=ax[2])
    plt.colorbar(img3, ax=ax[3])
    plt.colorbar(img4, ax=ax[4])
    # ax[0].set_title("co2 | A:LBS | sb average | all obsids | Feed 1\nBinned with noise weighting")
    ax[0].set_title("Binned with noise weighting")
    ax[1].set_title("Destriped")
    ax[2].set_title("Destriper template only")
    ax[3].set_title("El Template only")
    ax[4].set_title("Nhits")

    for i in range(4):
        ax[i].set_xlabel("Azimuth [degrees]")
        ax[i].set_ylabel("Elevation [degrees]")
    plt.tight_layout()
    plt.savefig("plots/" + filename + ".png", bbox_inches="tight")


if __name__ == "__main__":

    ### Parsing command line arguments. ###
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--field", type=str, default="co2", help="Chosen field.")
    parser.add_argument("-d", "--detectors", type=int, default=0, help="Number of detectors (feeds) to use.")
    parser.add_argument("-b", "--basissize", type=int, default=1000, help="Length of template basis vectors used for destriping.")
    parser.add_argument("-s", "--sideband", type=int, default=0, help="Sideband number to use.")
    parser.add_argument("-p", "--pixels", type=int, default=80, help="Number of map pixels in el and az.")
    parser.add_argument("-n", "--numeroffiles", type=int, default=10000, help="Number of files to use.")
    parser.add_argument("-a", "--azimuth", type=str, default="(0,360)", help="Azimuth range to use. Must be passed as tuple-like string.")
    parser.add_argument("-e", "--elevation", type=str, default="(30,60)", help="Elevation range to use. Must be passed as tuple-like string.")
    parser.add_argument("-c", "--scantype", type=str, default="all", help="To use all scans, CES scans, or Liss scans.")

    args = parser.parse_args()
    selected_field = args.field
    selected_sb = args.sideband
    selected_feeds = args.detectors
    selected_basis_size = args.basissize
    selected_pixels = args.pixels
    selected_files = args.numeroffiles
    az_range = eval(args.azimuth)
    el_range = eval(args.elevation)
    scantype = args.scantype
    if not selected_field in ["co2", "co6", "co7"]:
        raise ValueError("Do not recognize your chosen field.")
    if not selected_sb in range(4):
        raise ValueError("Sideband out of range.")
    if not selected_feeds in range(20):
        raise ValueError("Feeds out of range.")
    if not selected_pixels > 10:
        raise ValueError("Too few pixels.")
    if not selected_files > 1:
        raise ValueError("Too few selected files.")
    if not scantype in ["all", "CES", "Liss"]:
        raise ValueError("Unknown scan type.")
    if scantype == "all":
        scantypes = [2**4, 2**5, 2**15]
    elif scantype == "CES":
        scantypes = [2**5]
    elif scantype == "Liss":
        scantypes = [2**15]

    ### Reading obsid/scan info into dictionary. ###
    obsid_dict = {}
    with open(f"/mn/stornext/d16/cmbco/comap/protodir/runlist_default_{selected_field}.txt", "r") as infile:
        infile.readline()
        field, Nobsids = infile.readline().split()
        Nobsids = int(Nobsids)
        print(field, Nobsids)
        for obsid_idx in range(Nobsids):
            temp_obsid = infile.readline().split()
            obsid = temp_obsid[0]
            Nscans = int(temp_obsid[3])
            if Nscans != 0:
                obsid_dict[obsid] = []
                for scan_idx in range(Nscans):
                    temp_scan = infile.readline().split()
                    t0 = float(temp_scan[1])
                    t1 = float(temp_scan[2])
                    scan_type = int(temp_scan[3])
                    if scan_type in scantypes:
                        obsid_dict[obsid].append([t0,t1])
                if len(obsid_dict[obsid]) < 1:
                    obsid_dict.pop(obsid)
                elif (int(obsid) < OBSID_RANGE[0]) or (int(obsid) > OBSID_RANGE[1]):
                    obsid_dict.pop(obsid)

    print("Obsid dict size: ", len(obsid_dict))
    ### Finding all relevant filenames. ###
    filenames = []
    with h5py.File("../level1_catalogue.hd5", "r") as f:
        sources = np.array(f["sources"])
        filenames = np.array(f["filepaths"])

    idxs = sources == selected_field
    filenames = filenames[idxs]
    selected_files = min(selected_files, len(filenames))
    filenames = filenames[:selected_files]
    print("Number of files = ", len(filenames))

    
    ### Reading TOD data from all chosen files. ###
    TOD = TODhandler(el_range=el_range, az_range=az_range, Nfeeds=selected_feeds, sb=selected_sb)
    TOD.load_data_from_files(filenames)

    np.save("data/tod.npy", TOD.tod)
    np.save("data/el.npy", TOD.el)
    np.save("data/obsid_idxs.npy", TOD.obsid_idxs)
    
    ### Reading TOD arrays into mapmaker. ###
    mapmaker = Mapmaker()
    mapmaker.TOD_from_arrays(TOD, samprate=1.0/0.02, Nsidemap=selected_pixels)

    ### Running mapmaker. ###    
    mapmaker.make_basis(basis_type="constant_plus_el_per_obsid", basis_size=selected_basis_size)
    mapmaker.map_destripe()

    ### Reading maps. ###
    map_destripe = mapmaker.map_destripe.reshape(mapmaker.Nsidemap, mapmaker.Nsidemap)
    map_hits = mapmaker.hitmap.reshape(mapmaker.Nsidemap, mapmaker.Nsidemap)
    map_nw = mapmaker.map_nw.reshape(mapmaker.Nsidemap, mapmaker.Nsidemap)
    map_full_template = mapmaker.template_map.reshape(mapmaker.Nsidemap, mapmaker.Nsidemap)
    map_el_template = mapmaker.el_template_map.reshape(mapmaker.Nsidemap, mapmaker.Nsidemap)

    ### Writing maps to file. ###
    Nsidemap = mapmaker.Nsidemap
    filename = f"maps_{selected_field}_sb{selected_sb}_f{selected_feeds}_b{selected_basis_size}_p{selected_pixels}_n{selected_files}_WN{WN_INTERVAL}_O{OBSID_RANGE}_a{az_range}_e{el_range}_tol{TOL}_s{scantype}"

    with h5py.File(f"data/{filename}.hd5", "w") as f:
        f.create_dataset("el_range", (Nsidemap,), np.float64, np.linspace(mapmaker.el.min(), mapmaker.el.max(), mapmaker.Nsidemap))
        f.create_dataset("az_range", (Nsidemap,), np.float64, np.linspace(mapmaker.az.min(), mapmaker.az.max(), mapmaker.Nsidemap))
        f.create_dataset("map_destripe", (Nsidemap,Nsidemap), np.float64, map_destripe)
        f.create_dataset("map_hits", (Nsidemap,Nsidemap), np.float64, map_hits)
        f.create_dataset("map_nw", (Nsidemap,Nsidemap), np.float64, map_nw)
        f.create_dataset("map_full_template", (Nsidemap,Nsidemap), np.float64, map_full_template)
        f.create_dataset("map_el_template", (Nsidemap,Nsidemap), np.float64, map_el_template)
        f.create_dataset("a", (mapmaker.Nbasis,), np.float64, mapmaker.a)
        f.create_dataset("obsids", (len(TOD.obsids),), np.int64, TOD.obsids)
    
    make_plots(filename)
