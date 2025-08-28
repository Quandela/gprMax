from gprMax.fields_updates_ext import update_electric_cyl, update_magnetic_cyl
import numpy as np

dr = 0.1e-6
dz = 0.1e-6
m  = 2
nthreads = 1
nz = 1
nphi = 1
nr = 1
Er = [0]
Ephi = [0]
Ez = [1]
Hr = [1]
Hphi = [0]
Hz = [0]

update_coeffsE = np.array([[1.0000000e+00, 2.6638855e+02, 1.0000000e+00, 2.6638855e+02, 2.6638856e-05] for _ in range(6)])
update_coeffsH = np.array([[1.0000000e+00, 1.8769575e-03, 1.0000000e+00, 1.8769575e-03, 1.8769575e-10] for _ in range(6)])

ID = np.array([[[[1 for _ in range(nz)] for _ in range(nphi)] for _ in range(nr)] for _ in range(6)])

update_electric_cyl(nr, nz, m, nthreads, dr, dz, update_coeffsE, ID, Er, Ephi, Ez, Hr, Hphi, Hz)