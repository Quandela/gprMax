from gprMax.constants import floattype_t, complextype_t
cimport numpy as np

cpdef void update_electric(
    floattype_t dr,
    floattype_t dphi,
    floattype_t dz,
    int nr,
    int nphi,
    int nz,
    floattype_t[:, :, ::1] coeffsupdateE,
    floattype_t[:, : ,::1] coeffsupdateH,
    complextype_t[:, :, ::1] Er,
    complextype_t[:, :, ::1] Ephi,
    complextype_t[:, :, ::1] Ez,
    complextype_t[:, :, ::1] Hr,
    complextype_t[:, :, ::1] Hphi,
    complextype_t[:, :, ::1] Hz,
    complextype_t[::1] sum_Hphi,
    int m,
    int nthreads
):
    cdef Py_ssize_t i, j, k, j_update
    if dphi != 0:
        #k = 0, j = 0, r = 0
        materialEr = ID[0,0,0,0]
        materialEphi = ID[1,0,0,0]
        materialEz = ID[2,0,0,0]
        Er[0,0,0] = coeffsupdateE[materialEr, 0] * Er[0,0,0] + coeffsupdateE[materialEr, 2] * (Hz[0,0,0] - Hz[0,nphi-1,0]) / (0.5) - updatecoeffsE[materialEr, 3]*(Hphi[0,0,0] - 0)
        Ephi[0,0,0] = coeffsupdateE[materialEphi, 0] * Ephi[0,0,0] + coeffsupdateE[materialEphi, 3] * (Hr[0,0,0] - 0) - coeffsupdateE[materialEphi, 1] * (Hz[0,0,0] - Hz[0, nphi//2, 0])
        Ez[0,0,0] = coeffsupdateE[materialEz, 0] * Ez[0,0,0] + coeffsupdateE[materialEz, 1] * sum_Hphi[k]

        #First, let's take care of iterations involving r=0 and j=0
        for k in prange(1, nz, nogil= True, schedule= 'static', num_threads= nthreads):
            materialEr = ID[0,0,0,k]
            materialEphi = ID[1,0,0,k]
            materialEz = ID[2,0,0,k]
            Er[0,0,k] = coeffsupdateE[materialEr, 0] * Er[0,0,k] + coeffsupdateE[materialEr, 2] * (Hz[0,0,k] - Hz[0,nphi-1,k]) / (0.5) - updatecoeffsE[materialEr, 3]*(Hphi[0,0,k] - Hphi[0,0,k-1])
            Ephi[0,0,k] = coeffsupdateE[materialEphi, 0] * Ephi[0,0,k] + coeffsupdateE[materialEphi, 3] * (Hr[0,0,k] - Hr[0,0,k-1]) - coeffsupdateE[materialEphi, 1] * (Hz[0,0,k] - Hz[0, nphi//2, k])
            Ez[0,0,k] = coeffsupdateE[materialEz, 0] * Ez[0,0,k] + coeffsupdateE[materialEz, 1] * sum_Hphi[k]

        #Then with r = 0 and j != 0
        for j in prange(1, nr, nogil= True, schedule= 'static', num_threads= nthreads):
            #k = 0
            materialEr = ID[0,0,j,0]
            materialEphi = ID[1,0,j,0]
            materialEz = ID[2,0,j,0]
            Er[0,j,0] = coeffsupdateE[materialEr, 0] * Er[0,j,0] + coeffsupdateE[materialEr, 2] * (Hz[0,j,0] - Hz[0,j-1,0]) / (0.5) - updatecoeffsE[materialEr, 3]*(Hphi[0,j,0] - 0)
            Ephi[0,j,0] = coeffsupdateE[materialEphi, 0] * Ephi[0,j,0] + coeffsupdateE[materialEphi, 3] * (Hr[0,j,k] - 0) - coeffsupdateE[materialEphi, 1] * (Hz[0,j,0] - Hz[0, j_update, 0])
            Ez[0,j,0] = coeffsupdateE[materialEz, 0] * Ez[0,j,0] + coeffsupdateE[materialEz, 1] * sum_Hphi[0]
            j_update = (j + <int> nphi // 2)% <int> nphi

            #k != 0, j != 0, r = 0
            for k in range(1,nz):
                materialEr = ID[0,0,j,k]
                materialEphi = ID[1,0,j,k]
                materialEz = ID[2,0,j,k]
                Er[0,j,k] = coeffsupdateE[materialEr, 0] * Er[0,j,k] + coeffsupdateE[materialEr, 2] * (Hz[0,j,k] - Hz[0,j-1,k]) / (0.5) - updatecoeffsE[materialEr, 3]*(Hphi[0,j,k] - Hphi[0,j,k-1])
                Ephi[0,j,k] = coeffsupdateE[materialEphi, 0] * Ephi[0,j,k] + coeffsupdateE[materialEphi, 3] * (Hr[0,j,k] - Hr[0,j,k-1]) - coeffsupdateE[materialEphi, 1] * (Hz[0,j,k] - Hz[0, j_update, k])
                Ez[0,j,k] = coeffsupdateE[materialEz, 0] * Ez[0,j,k] + coeffsupdateE[materialEz, 1] * sum_Hphi[k]
        
        # And now, r != 0
        for i in prange(1, nr, nogil= True, schedule= 'static', num_threads= nthreads):
            # k = 0, j = 0
            materialEr = ID[0,i,0,0]
            materialEphi = ID[1,i,0,0]
            materialEz = ID[2,i,0,0]
            Er[i,0,0] = updatecoeffsE[materialEr, 0] * Er[i,0,0] + coeffsupdateE[materialEr, 2]*(Hz[i,0,k] - Hz[i,nphi-1,0])/(i+0.5) - coeffsupdateE[materialEr, 3] * Hphi[i,0,0]
            Ephi[i,0,0] = updatecoeffsE[materialEphi, 0] * Ephi[i,0,0] + updatecoeffsE[materialEphi, 3] * Hr[i,0,0] - updatecoeffsE[materialEphi, 1] * (Hz[i,0,0] - Hz[i-1,0,0])
            Ez[i,0,0] = updatecoeffsE[materialEz, 0] * Ez[i,0,0] + updatecoeffsE[materialEz, 1] * ((i+0.5) * Hphi[i,0,0] - (i-0.5) * Hphi[i-1,0,0])/i - updatecoeffsE[materialEz, 2]*(Hr[i,0,0] - Hr[i,nphi-1,0])/i
            # k = 0, j != 0
            for j in range(1, nphi):
                materialEr = ID[0,i,j,0]
                materialEphi = ID[1,i,j,0]
                materialEz = ID[2,i,j,0]
                Er[i,j,0] = updatecoeffsE[materialEr, 0] * Er[i,j,0] + coeffsupdateE[materialEr, 2]*(Hz[i,j,0] - Hz[i,j-1,0])/(i+0.5) - coeffsupdateE[materialEr, 3] * Hphi[i,j,0]
                Ephi[i,j,0] = updatecoeffsE[materialEphi, 0] * Ephi[i,j,0] + updatecoeffsE[materialEphi, 3] * Hr[i,j,0] - updatecoeffsE[materialEphi, 1] * (Hz[i,j,0] - Hz[i-1,j,0])
                Ez[i,j,0] = updatecoeffsE[materialEz, 0] * Ez[i,j,0] + updatecoeffsE[materialEz, 1] * ((i+0.5) * Hphi[i,j,0] - (i-0.5) * Hphi[i-1,j,0])/i - updatecoeffsE[materialEz, 2]*(Hr[i,j,0] - Hr[i,j-1,0])/i
            # j = 0, k != 0
            for k in range(1, nz):
                materialEr = ID[0,i,0,k]
                materialEphi = ID[1,i,0,k]
                materialEz = ID[2,i,0,k]
                Er[i,0,k] = updatecoeffsE[materialEr, 0] * Er[i,0,k] + coeffsupdateE[materialEr, 2]*(Hz[i,0,k] - Hz[i,nphi-1,k])/(i+0.5) - coeffsupdateE[materialEr, 3] * Hphi[i,0,k]
                Ephi[i,0,k] = updatecoeffsE[materialEphi, 0] * Ephi[i,0,k] + updatecoeffsE[materialEphi, 3] * Hr[i,0,k] - updatecoeffsE[materialEphi, 1] * (Hz[i,0,k] - Hz[i-1,0,k])
                Ez[i,0,k] = updatecoeffsE[materialEz, 0] * Ez[i,0,k] + updatecoeffsE[materialEz, 1] * ((i+0.5) * Hphi[i,0,k] - (i-0.5) * Hphi[i-1,0,k])/i - updatecoeffsE[materialEz, 2]*(Hr[i,0,k] - Hr[i,nphi-1,k])/i
            # j != 0, k != 0
            for j in range(1, nphi):
                for k in range(1, nz):  
                    materialEr = ID[0,i,j,k]
                materialEphi = ID[1,i,j,k]
                materialEz = ID[2,i,j,k]
                Er[i,j,k] = updatecoeffsE[materialEr, 0] * Er[i,j,k] + coeffsupdateE[materialEr, 2]*(Hz[i,j,k] - Hz[i,j-1,k])/(i+0.5) - coeffsupdateE[materialEr, 3] * Hphi[i,j,k]
                Ephi[i,j,k] = updatecoeffsE[materialEphi, 0] * Ephi[i,j,k] + updatecoeffsE[materialEphi, 3] * Hr[i,j,k] - updatecoeffsE[materialEphi, 1] * (Hz[i,j,k] - Hz[i-1,j,k])
                Ez[i,j,k] = updatecoeffsE[materialEz, 0] * Ez[i,j,k] + updatecoeffsE[materialEz, 1] * ((i+0.5) * Hphi[i,j,k] - (i-0.5) * Hphi[i-1,j,k])/i - updatecoeffsE[materialEz, 2]*(Hr[i,j,k] - Hr[i,j-1,k])/i

    else:
        raise GeneralError("Cylindrical symmetry not yet implemented")