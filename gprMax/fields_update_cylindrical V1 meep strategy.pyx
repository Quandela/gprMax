###################################
# Cylindrical coordinates updates #
###################################

from libc.stdlib cimport malloc, free
from libc cimport complex
from cython cimport boundscheck, wraparound, nonecheck

cdef extern from "complex.h":
    double complex I

cdef void free_complex3D(double complex*** arr, Py_ssize_t nx, Py_ssize_t ny):
    cdef Py_ssize_t i, j
    for i in range(nx):
        for j in range(ny):
            free(arr[i][j])
        free(arr[i])
    free(arr)

cdef double complex*** alloc_and_copy_complex3D(np.complex128_t[:, :, ::1] arr):
    """
    Alloue un tableau 3D C (double complex***) et copie les données d'un memoryview NumPy.

    Args:
        arr (np.complex128_t[:, :, ::1]): Tableau NumPy d'entrée

    Returns:
        double complex*** : Tableau C alloué et rempli
    """
    cdef Py_ssize_t nx = arr.shape[0]
    cdef Py_ssize_t ny = arr.shape[1]
    cdef Py_ssize_t nz = arr.shape[2]

    cdef double complex*** out
    cdef Py_ssize_t i, j, k

    # Allocation des pointeurs
    out = <double complex***> malloc(nx * sizeof(double complex**))
    for i in range(nx):
        out[i] = <double complex**> malloc(ny * sizeof(double complex*))
        for j in range(ny):
            out[i][j] = <double complex*> malloc(nz * sizeof(double complex))

    # Copie des données
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                out[i][j][k] = arr[i, j, k]

    return out

cdef void copy_complex3D_to_numpy(double complex*** src,
                                       np.complex128_t[:, :, ::1] dest):
    """
    Copie les données d’un tableau C (double complex***) dans un memoryview NumPy,
    sans avoir besoin de spécifier les dimensions.

    Args:
        src : tableau C (double complex***)
        dest : memoryview NumPy déjà alloué
    """
    cdef Py_ssize_t nx = dest.shape[0]
    cdef Py_ssize_t ny = dest.shape[1]
    cdef Py_ssize_t nz = dest.shape[2]
    cdef Py_ssize_t i, j, k

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                dest[i, j, k] = src[i][j][k]


cpdef void update_electric_cyl(
                    int nr,
                    int nz,
                    int m,
                    int nthreads,
                    float dr,
                    float dz,
                    float eps,
                    foat sigma,
                    floattype_t[:, ::1] updatecoeffsE, #Courant factor is already included in dt
                    np.uint32_t[:, :, :, ::1] ID,
                    np.complex128_t[:, :, ::1] Er_np,
                    np.complex128_t[:, :, ::1] Ephi_np,
                    np.complex128_t[:, :, ::1] Ez_np,
                    np.complex128_t[:, :, ::1] Hr_np,
                    np.complex128_t[:, :, ::1] Hphi_np,
                    np.complex128_t[:, :, ::1] Hz_np
            ):
    """This function updates the electric field components.

    Args:
        nr, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """

    cdef Py_ssize_t i, k, taille_i_fields, taille_j_fields
    cdef int materialEr, materialEphi, materialEz
    # It may seem like we are updated inside the pmls, and we are ! But those fields will be overwritten by the update
    # of the pml fields after this function

    cdef double complex*** Er = alloc_and_copy_complex3D(Er_np)
    cdef double complex*** Ephi = alloc_and_copy_complex3D(Ephi_np)
    cdef double complex*** Ez = alloc_and_copy_complex3D(Ez_np)
    cdef double complex*** Hr = alloc_and_copy_complex3D(Hr_np)
    cdef double complex*** Hphi = alloc_and_copy_complex3D(Hphi_np)
    cdef double complex*** Hz = alloc_and_copy_complex3D(Hz_np)

    taille_i_fields = Er_np.shape[0]
    taille_j_fields = Er_np.shape[1]

    for i in prange(0, nr, nogil= True, schedule= 'static', num_threads= nthreads):
        for k in range(0, nz):
            materialEr = ID[0,i,0,k]
            materialEphi = ID[1,i,0,k]
            materialEz = ID[2,i,0,k]

            Er[i][0][k] = Er[i][0][k] * updatecoeffsE[materialEr, 0] + I * m / (i+0.5) * Hz[i][0][k] * updatecoeffsE[materialEr, 1] - (Hphi[i][0][k] - Hphi[i][0][k-1]) * updatecoeffsE[materialEr, 3]

            Ephi[i][0][k] = Ephi[i][0][k] * updatecoeffsE[materialEphi, 0] + (Hr[i][0][k] - Hr[i][0][k-1]) * updatecoeffsE[materialEphi,3] - (Hz[i][0][k] - Hz[i-1][0][k]) * updatecoeffsE[materialEphi, 1]

            Ez[i][0][k] = Ez[i][0][k] * updatecoeffsE[materialEz, 0] + (Hphi[i][0][k]*(i+0.5) - Hphi[i-1][0][k]*(i-0.5))/i * updatecoeffsE[materialEz, 1] - I * m / i * Hr[i][0][k] * updatecoeffsE[materialEz, 1]

    copy_complex3D_to_numpy(Er, Er_np)
    copy_complex3D_to_numpy(Ephi, Ephi_np)
    copy_complex3D_to_numpy(Ez, Ez_np)
    copy_complex3D_to_numpy(Hr, Hr_np)
    copy_complex3D_to_numpy(Hphi, Hphi_np)
    copy_complex3D_to_numpy(Hz, Hz_np)

    free_complex3D(Er,taille_i_fields, taille_j_fields)
    free_complex3D(Ephi,taille_i_fields, taille_j_fields)
    free_complex3D(Ez,taille_i_fields, taille_j_fields)
    free_complex3D(Hr,taille_i_fields, taille_j_fields)
    free_complex3D(Hphi,taille_i_fields, taille_j_fields)
    free_complex3D(Hz,taille_i_fields, taille_j_fields)

cpdef void update_magnetic_cyl(
                    int nr,
                    int nz,
                    int m,
                    int nthreads,
                    floattype_t[:, ::1] updatecoeffsH,
                    np.uint32_t[:, :, :, ::1] ID,
                    np.complex128_t[:, :, ::1] Er_np,
                    np.complex128_t[:, :, ::1] Ephi_np,
                    np.complex128_t[:, :, ::1] Ez_np,
                    np.complex128_t[:, :, ::1] Hr_np,
                    np.complex128_t[:, :, ::1] Hphi_np,
                    np.complex128_t[:, :, ::1] Hz_np
            ):
    """This function updates the electric field components.

    Args:
        nr, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """

    cdef Py_ssize_t i, k, taille_i_fields, taille_j_fields
    cdef int materialHr, materialHphi, materialHz

    cdef double complex*** Er = alloc_and_copy_complex3D(Er_np)
    cdef double complex*** Ephi = alloc_and_copy_complex3D(Ephi_np)
    cdef double complex*** Ez = alloc_and_copy_complex3D(Ez_np)
    cdef double complex*** Hr = alloc_and_copy_complex3D(Hr_np)
    cdef double complex*** Hphi = alloc_and_copy_complex3D(Hphi_np)
    cdef double complex*** Hz = alloc_and_copy_complex3D(Hz_np)

    taille_i_fields = Er_np.shape[0]
    taille_j_fields = Er_np.shape[1]

    for i in prange(1, nr, nogil= True, schedule= 'static', num_threads= nthreads):
        for k in range(1, nz):
            materialHr = ID[3,i,0,k]
            materialHphi = ID[4,i,0,k]
            materialHz = ID[5,i,0,k]

            Hr[i][0][k] = Hr[i][0][k] * updatecoeffsH[materialHr, 0] + (Ephi[i][0][k+1] - Ephi[i][0][k]) * updatecoeffsH[materialHr, 3] - I * m / i * Ez[i+1][0][k] * updatecoeffsH[materialHr, 1] 

            Hphi[i][0][k] = Hphi[i][0][k] * updatecoeffsH[materialHphi, 0] + (Ez[i+1][0][k] - Ez[i][0][k]) * updatecoeffsH[materialHphi,1] - (Er[i][0][k+1] - Er[i][0][k]) * updatecoeffsH[materialHphi, 3]

            Hz[i][0][k] = Hz[i][0][k] * updatecoeffsH[materialHz, 0] + I * m / (i+0.5) * Er[i][0][k] * updatecoeffsH[materialHz, 1] - (Ephi[i+1][0][k]*(i+1) - Ephi[i][0][k]*i)/(i+0.5) * updatecoeffsH[materialHz, 1]

    copy_complex3D_to_numpy(Er, Er_np)
    copy_complex3D_to_numpy(Ephi, Ephi_np)
    copy_complex3D_to_numpy(Ez, Ez_np)
    copy_complex3D_to_numpy(Hr, Hr_np)
    copy_complex3D_to_numpy(Hphi, Hphi_np)
    copy_complex3D_to_numpy(Hz, Hz_np)  

    free_complex3D(Er,taille_i_fields, taille_j_fields)
    free_complex3D(Ephi,taille_i_fields, taille_j_fields)
    free_complex3D(Ez,taille_i_fields, taille_j_fields)
    free_complex3D(Hr,taille_i_fields, taille_j_fields)
    free_complex3D(Hphi,taille_i_fields, taille_j_fields)
    free_complex3D(Hz,taille_i_fields, taille_j_fields)

cpdef update_magnetic_origin(
                    int nr,
                    int nz,
                    int m,
                    int nthreads,
                    floattype_t dt,
                    floattype_t mu,
                    floattype_t dr,
                    floattype_t[:, ::1] updatecoeffsH,
                    floattype_t[:, ::1] updatecoeffsE,
                    np.uint32_t[:, :, :, ::1] ID,
                    np.complex128_t[:, :, ::1] Er_np,
                    np.complex128_t[:, :, ::1] Ephi_np,
                    np.complex128_t[:, :, ::1] Ez_np,
                    np.complex128_t[:, :, ::1] Hr_np,
                    np.complex128_t[:, :, ::1] Hphi_np,
                    np.complex128_t[:, :, ::1] Hz_np
):
    
    cdef Py_ssize_t i, k, taille_i_fields, taille_j_fields
    cdef int materialHr, materialHphi, materialHz

    cdef double complex*** Er = alloc_and_copy_complex3D(Er_np)
    cdef double complex*** Ephi = alloc_and_copy_complex3D(Ephi_np)
    cdef double complex*** Ez = alloc_and_copy_complex3D(Ez_np)
    cdef double complex*** Hr = alloc_and_copy_complex3D(Hr_np)
    cdef double complex*** Hphi = alloc_and_copy_complex3D(Hphi_np)
    cdef double complex*** Hz = alloc_and_copy_complex3D(Hz_np)

    taille_i_fields = Er_np.shape[0]
    taille_j_fields = Er_np.shape[1]

    cdef floattype_t pi = np.pi
    cdef floattype_t e = np.exp(1)
    cdef int Nphi = int(nr//4 * 4)
    print("update_magnetic_origin 1")

    for k in prange(1, nz, nogil=True, schedule= 'static', num_threads= nthreads):
        Hr[0][0][k] = updatecoeffsE[ID[1,0,0,k],0] * Hr[0][0][k] + 0.5 * updatecoeffsE[ID[1,0,0,k], 1] * (e**(I*m*pi* 3 / 4 * Nphi/nr) - e**(I*m*pi / 4 * Nphi/nr)) * Ez[1][0][k] + 0.5 * updatecoeffsE[ID[1,0,0,k], 3] * (- e**(I * m * pi * 3 * Nphi / nr) + e**(I * m * pi * Nphi / nr)) * Ephi[0][0][k+1] + 0.5 * updatecoeffsE[ID[1,0,0,k], 3] * (e**(I * m * pi * 3 * Nphi / nr) - e**(I * m * pi * Nphi / nr)) * Ephi[0][0][k]
        Hphi[0][0][k] = Hphi[0][0][k] * updatecoeffsH[ID[4,0,0,k], 0] + (Ez[1][0][k] - Ez[0][0][k]) * updatecoeffsH[ID[4,0,0,k],1] - (Er[0][0][k+1] - Er[0][0][k]) * updatecoeffsH[ID[4,0,0,k], 3]
        Hz[0][0][k] += Hz[0][0][k] * updatecoeffsH[ID[5,0,0,k], 0] + dt / mu * (Er[0][0][k] * I * m - Ephi[1][0][k]) / (0.5 * dr)
    print("update_magnetic_origin 2")

    copy_complex3D_to_numpy(Hr, Hr_np)
    copy_complex3D_to_numpy(Hphi, Hphi_np)
    copy_complex3D_to_numpy(Hz, Hz_np)  

    free_complex3D(Er,taille_i_fields, taille_j_fields)
    free_complex3D(Ephi,taille_i_fields, taille_j_fields)
    free_complex3D(Ez,taille_i_fields, taille_j_fields)
    free_complex3D(Hr,taille_i_fields, taille_j_fields)
    free_complex3D(Hphi,taille_i_fields, taille_j_fields)
    free_complex3D(Hz,taille_i_fields, taille_j_fields)

cpdef update_electric_origin(
                    int nr,
                    int nz,
                    int m,
                    int nthreads,
                    floattype_t dt,
                    floattype_t mu,
                    floattype_t dr,
                    floattype_t[:, ::1] updatecoeffsH,
                    floattype_t[:, ::1] updatecoeffsE,
                    np.uint32_t[:, :, :, ::1] ID,
                    np.complex128_t[:, :, ::1] Er_np,
                    np.complex128_t[:, :, ::1] Ephi_np,
                    np.complex128_t[:, :, ::1] Ez_np,
                    np.complex128_t[:, :, ::1] Hr_np,
                    np.complex128_t[:, :, ::1] Hphi_np,
                    np.complex128_t[:, :, ::1] Hz_np
):
    
    cdef Py_ssize_t j, k, taille_i_fields, taille_j_fields
    cdef int materialEr, materialEphi, materialEz

    cdef double complex*** Er = alloc_and_copy_complex3D(Er_np)
    cdef double complex*** Ephi = alloc_and_copy_complex3D(Ephi_np)
    cdef double complex*** Ez = alloc_and_copy_complex3D(Ez_np)
    cdef double complex*** Hr = alloc_and_copy_complex3D(Hr_np)
    cdef double complex*** Hphi = alloc_and_copy_complex3D(Hphi_np)
    cdef double complex*** Hz = alloc_and_copy_complex3D(Hz_np)

    cdef floattype_t pi = np.pi
    cdef floattype_t e = np.exp(1)
    cdef int Nphi = int(nr//4 * 4)
    cdef floattype_t m_coeff

    taille_i_fields = Er_np.shape[0]
    taille_j_fields = Er_np.shape[1]
    print("update_electric_origin 1")
    for k in prange(1, nz, nogil=True, schedule= 'static', num_threads= nthreads):
        m_coeff = 4*updatecoeffsE[ID[2,0,0,k], 1] / Nphi
        Er[0][0][k] = Er[0][0][k] * updatecoeffsE[ID[1,0,0,k], 0] + I * m / (0.5) * Hz[0][0][k] * updatecoeffsE[ID[1,0,0,k], 1] - (Hphi[0][0][k] - Hphi[0][0][k-1]) * updatecoeffsE[ID[1,0,0,k], 3]
        with gil:
            print("Passage 1 safe")
        Ephi[0][0][k] = updatecoeffsE[ID[1,0,0,k], 0] * Ephi[0][0][k] + (Hr[0][0][k] - Hr[0][0][k-1]) * updatecoeffsE[ID[1,0,0,k], 3] - Hz[1][0][k] * e**(I * pi * Nphi / nr) * updatecoeffsE[ID[1,0,0,k], 1]
        with gil:
            print("Passage 2 safe")
        Ez[0][0][k] *= updatecoeffsE[ID[2,0,0,k], 0] * Ez[0][0][k]
        with gil:
            print("Passage 3 safe")
        for j in range(1, Nphi):
            Ez[0][0][k] += m_coeff * Hphi[1][0][k+1] * e**(I * m * pi * j / Nphi )
    print("update_electric_origin 2")

    copy_complex3D_to_numpy(Er, Er_np)
    copy_complex3D_to_numpy(Ephi, Ephi_np)
    copy_complex3D_to_numpy(Ez, Ez_np)
    copy_complex3D_to_numpy(Hr, Hr_np)
    copy_complex3D_to_numpy(Hphi, Hphi_np)
    copy_complex3D_to_numpy(Hz, Hz_np)

    free_complex3D(Er,taille_i_fields, taille_j_fields)
    free_complex3D(Ephi,taille_i_fields, taille_j_fields)
    free_complex3D(Ez,taille_i_fields, taille_j_fields)
    free_complex3D(Hr,taille_i_fields, taille_j_fields)
    free_complex3D(Hphi,taille_i_fields, taille_j_fields)
    free_complex3D(Hz,taille_i_fields, taille_j_fields)