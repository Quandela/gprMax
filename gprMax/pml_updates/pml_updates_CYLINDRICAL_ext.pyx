# Copyright (C) 2025: Quandela
#                 Authors: Quentin David
#
# This file is added to the modified code of gprMax allowing for cylindrical coordinate.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU GenRAl Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU GenRAl Public License for more details.
#
# You should have received a copy of the GNU GenRAl Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.
#
#To get information as to why the PMLs are updated the way they are, please
# check https://repository.mines.edu/server/api/core/bitstreams/2cbef3b2-38af-4c25-bb9c-55bdd3abec74/content

import numpy as np
cimport numpy as np
from cython.parallel import prange

from gprMax.constants cimport floattype_t
#from scipy.constants import epsilon_0 as e0
#from scipy.constants import mu_0 as mu0

from libc.stdio cimport printf

cdef extern from "complex.h":
    double complex I

cdef double mu0 = 4 * 3.141592653589793 * 1e-7
cdef double e0 = 8.854187817e-12  
cdef double e = 2.718281828459045



################## Conversion functions needed as we can't use operations on np.complex without gil ############################

from libc.stdlib cimport malloc, free
from libc cimport complex
from cython cimport boundscheck, wraparound, nonecheck



cdef extern from "complex.h":
    pass  # pour éviter certains warnings avec Cython


@boundscheck(False)
@wraparound(False)
@nonecheck(False)
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


@boundscheck(False)
@wraparound(False)
@nonecheck(False)
cdef double complex**** alloc_and_copy_complex4D(np.complex128_t[:, :, :, ::1] arr):
    """
    Alloue un tableau 4D C (double complex***) et copie les données d'un memoryview NumPy.

    Args:
        arr (np.complex128_t[:, :, ::1]): Tableau NumPy d'entrée

    Returns:
        double complex*** : Tableau C alloué et rempli
    """
    cdef Py_ssize_t nx = arr.shape[0]
    cdef Py_ssize_t ny = arr.shape[1]
    cdef Py_ssize_t nz = arr.shape[2]
    cdef Py_ssize_t n4 = arr.shape[3]

    cdef double complex**** out
    cdef Py_ssize_t i, j, k, l

    # Allocation des pointeurs
    out = <double complex****> malloc(nx * sizeof(double complex**))
    for i in range(nx):
        out[i] = <double complex***> malloc(ny * sizeof(double complex*))
        for j in range(ny):
            out[i][j] = <double complex**> malloc(nz * sizeof(double complex))
            for k in range(nz):
                out[i][j][k] = <double complex*> malloc(n4 * sizeof(double complex))

    # Copie des données
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for l in range(n4):
                    out[i][j][k][l] = arr[i, j, k, l]

    return out

cdef void free_complex3D(double complex*** arr, Py_ssize_t nx, Py_ssize_t ny):
    cdef Py_ssize_t i, j
    for i in range(nx):
        for j in range(ny):
            free(arr[i][j])
        free(arr[i])
    free(arr)

cdef void free_complex4D(double complex**** arr, Py_ssize_t nx, Py_ssize_t ny, Py_ssize_t nz):
    cdef Py_ssize_t i, j, k
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                free(arr[i][j][k])
            free(arr[i][j])
        free(arr[i])
    free(arr)


@boundscheck(False)
@wraparound(False)
@nonecheck(False)
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

@boundscheck(False)
@wraparound(False)
@nonecheck(False)
cdef void copy_complex4D_to_numpy(double complex**** src,
                                       np.complex128_t[:, :, :, ::1] dest):
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
    cdef Py_ssize_t n4 = dest.shape[3]
    cdef Py_ssize_t i, j, k, l

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                for l in range(n4):
                    dest[i, j, k, l] = src[i][j][k][l]

cdef double complex sum3D_i(floattype_t[:, :, ::1] arr, Py_ssize_t j, Py_ssize_t k) noexcept nogil:

    cdef Py_ssize_t i, nr
    cdef double complex out = 0

    nr = arr.shape[0]
    for i in range(0,nr):
        out += arr[i,j,k]
    return out

cdef double complex sum4D_i(double complex**** arr,Py_ssize_t i, Py_ssize_t j, Py_ssize_t k, int nl) noexcept nogil :

    cdef Py_ssize_t l
    cdef double complex out = 0

    for l in range(0,nl):
        out += arr[i][j][k][l]
    return out

cdef double complex sum_list_c(double complex**** arr, Py_ssize_t i, Py_ssize_t j, Py_ssize_t k, int n4) noexcept nogil:

    cdef Py_ssize_t l
    cdef double complex out = 0
    for l in range(n4):
        out += arr[i][j][k][l]
    return out

cdef double complex simple_sum(double complex* arr, int n) noexcept nogil:
    cdef double complex out = 0
    cdef Py_ssize_t i
    for i in range(n):
       out += arr[i]
    return out

cdef double complex* Hadamard_product_i(double complex**** X_arr, floattype_t[:, :, ::1] const, Py_ssize_t i, Py_ssize_t j, Py_ssize_t k, int n4) noexcept nogil:
    cdef double complex* out
    cdef Py_ssize_t l

    out = <double complex*> malloc(n4 * sizeof(double complex))
    for l in range(n4):
        out[l] = X_arr[i][j][k][l] * const[l, j, k]
    
    return out

################## Update of the PMLs in the r direction ###############################
cpdef void initialize_constant_lists_rho( #OK pour les fomules des constantes
                    int rs,
                    int rf,
                    int nz,
                    int thickness_r,
                    float dr,
                    float dt,
                    int nthreads,
                    floattype_t[:, :, ::1] sigma,
                    floattype_t[:, :, ::1] alpha,
                    floattype_t[:, :, ::1] kappa,
                    floattype_t[:, :, ::1] return_omega,
                    floattype_t[:, :, ::1] return_Ksi_list,
                    floattype_t[:, :, ::1] return_Lambda_list,
                    floattype_t[:, :, ::1] return_Psi_list,
                    floattype_t[:, :, ::1] return_Theta_list,
                    floattype_t[:, :, ::1] return_alpha,
                    floattype_t[:, :, ::1] return_R_list,
):
    """
    This function computes the Ksi_list and Lambda_list used in the PML updates. Only updated once !
    
    Args:
        rs, rf (int): position of the PML along the r-axis
        nz (int): number of cells along the z axis
        dr (float): spatial discretization
        dt (float): timestep
        nthreads (int): number of threads to use
        sigma, alpha, kappa (memoryview): PML lists
        return_omega (memoryview): return list with omega values
        return_Ksi_list (memoryview): return list with Ksi_list values.
        return_Lambda_list (memoryview): return list with Lambda_list values. 
        return_Psi_list (memoryview): return list with Psi_list values. 
        return_Theta_list (memoryview): return list with Theta_list values. 
        return_alpha (memoryview): return list with exp(-alpha * dt / e0) values. 
        return_R_list (memoryview): return list with R_list values. 

    """
    cdef Py_ssize_t i, k, ii
    cdef int nr
    cdef float arg, alpha_term, sh, exp

    nr = thickness_r

    for k in prange(0, nz, nogil= True, schedule= 'static', num_threads = nthreads):
        for i in range(0, nr):
            arg = alpha[i,0,k]*kappa[i,0,k] + sigma[i,0,k]
            alpha_term = alpha[i, 0, k] * dt / e0
            sh = (e**(alpha_term/2) - e**(-alpha_term/2))/2
            exp = e**(-alpha_term)
            
            return_alpha[i,0,k] = exp #OK
            return_omega[i, 0, k] = sigma[i, 0, k] * dr * (1 - exp) / alpha[i, 0, k] #OK
            return_Ksi_list[i,0,k] = sigma[i,0,k] * dr * sh / e0 #OK
            return_Lambda_list[i,0,k] = sigma[i,0,k] * (1 - e**(-arg*dt/(kappa[i,0,k]*e0))) / arg #OK
            return_Psi_list[i,0,k] = sigma[i,0,k] * (1-exp) / alpha[i,0,k] #OK
            return_Theta_list[i,0,k] = sigma[i,0,k] / e0 * sh #OK
            if i == 0:
                return_R_list[i,0,k] = 0
            else:
                return_R_list[i,0,k] = return_R_list[i-1,0,k] + sigma[i,0,k] * dr / e0 #OK

cdef void update_XQEphi_( #OK
        int rs,
        int nz,
        int nthreads,
        int thickness_r,
        double complex*** EPhi,
        double complex**** XQEphi_, #XQEphi_[i,j,k] donne la matrice XQEphi_ au point (i,j,k)
        floattype_t[:, :, ::1] Omega_term_list,
        floattype_t[:, :, ::1] alpha_term_list,
):
    """
    This function updates XQEphi_ from time n to time n+1
    
    Args:
        rs, rf (int): position of the PML along the r-axis
        nz (int): number of cells along the z axis 
        nthreads (int): number of threads to use
        EPhi, Omega_term_list, alpha_term_list (memoryview): lists required for the update. EPhi_ is taken at time n+1
        XQEphi_: list to be updated
    """
    cdef Py_ssize_t i, k, ii, iii
    cdef int nr

    nr = thickness_r

    for k in prange(0, nz, nogil = True, num_threads= nthreads, schedule= 'static'):
        for i in range(0, nr):
            ii = rs + i
            for iii in range(0,nr):
                XQEphi_[i][0][k][iii] = Omega_term_list[iii,0,k] * EPhi[ii][0][k] + XQEphi_[i][0][k][iii] * alpha_term_list[iii, 0, k]

cdef void update_XQEzs( #OK
        int rs,
        int nz,
        int nthreads,
        int thickness_r,
        double complex*** Ezs,
        double complex**** XQEzs, #XQEzs[i,j,k] donne la matrice XQEzs au point (i,j,k)
        floattype_t[:, :, ::1] Ksi_term_list,
        floattype_t[:, :, ::1] alpha_term_list,
):
    """
    This function updates XQEphi_ from time n to time n+1

    Args:
        rs, rf (int): position of the PML along the r-axis
        nz (int): number of cells along the z axis 
        nthreads (int): number of threads to use
        Ezs, Ksi_term_list, alpha_term_list (memoryview): lists required for the update. EPhi_ is taken at time n+1
        XQEzs: list to be updated
    """
    cdef Py_ssize_t i, k, iii
    cdef int nr

    nr = thickness_r

    for k in prange(0, nz, nogil = True, num_threads= nthreads, schedule= 'static'):
        for i in range(0, nr):
            for iii in range(0,nr):
                XQEzs[i][0][k][iii] = Ksi_term_list[iii][0][k] * Ezs[i][0][k] + XQEzs[i][0][k][iii] * alpha_term_list[iii, 0, k]

cdef void update_XQHphi_( #OK
        int rs,
        int nz,
        int nthreads,
        int thickness_r,
        double complex*** Hphi,
        double complex**** XQHphi_, #XQHphi[i,j,k] donne la matrice XQEzs au point (i,j,k)
        floattype_t[:, :, ::1] Omega_term_list,
        floattype_t[:, :, ::1] alpha_term_list,
):
    """
    This function updates XQHphi_ from time n-1/2 to time n+1/2

    Args:
        rs, rf (int): position of the PML along the r-axis
        nz (int): number of cells along the z axis 
        nthreads (int): number of threads to use
        HPhi, Omega_term_list (memoryview): lists required for the update. EPhi_ is taken at time n+1
        XQHphi_: list to be updated
    """
    cdef Py_ssize_t i, k, ii, iii
    cdef int nr

    nr = thickness_r

    for k in prange(0, nz, nogil = True, num_threads= nthreads, schedule= 'static'):
        for i in range(0, nr):
            ii = rs + i
            for iii in range(nr):
                XQHphi_[i][0][k][iii] = Omega_term_list[iii,0,k] * Hphi[ii][0][k] + XQHphi_[i][0][k][iii] * alpha_term_list[iii,0,k]

cdef void update_XQHzs( #OK
        int rs,
        int nz,
        int nthreads,
        int thickness_r,
        double complex*** Hzs,
        double complex**** XQHzs, #XQHzs[i,j,k] donne la matrice XQHzs au point (i,j,k)
        floattype_t[:, :, ::1] Ksi_term_list,
        floattype_t[:, :, ::1] alpha_term_list,
):
    """
    This function updates XQHzs from time n-1/2 to time n+1/2

    Args:
        rs, rf (int): position of the PML along the r-axis
        nz (int): number of cells along the z axis 
        nthreads (int): number of threads to use
        EPhi, Omega_term_list (memoryview): lists required for the update. EPhi_ is taken at time n+1
        XQEphi_: list to be updated
    """
    cdef Py_ssize_t i, k, iii
    cdef int nr

    nr = thickness_r
    for k in prange(0, nz, nogil = True, num_threads= nthreads, schedule= 'static'):
        for i in range(0, nr):
            for iii in range(nr):
                XQHzs[i][0][k][iii] = Ksi_term_list[iii,0,k] * Hzs[i][0][k] + XQHzs[i][0][k][iii] * alpha_term_list[iii,0,k]



cpdef void E_update_r_slab(
                        int rs,
                        int m,
                        int nz,
                        int thickness_r,
                        float dr,
                        float dz,
                        float dt,
                        int nthreads,
                        np.uint32_t[:, :, :, ::1] ID,
                        np.complex128_t[:, :, ::1] Er_np,
                        np.complex128_t[:, :, ::1] Ers_np,
                        np.complex128_t[:, :, ::1] QErs_np,
                        np.complex128_t[:, :, ::1] Ephi_np,
                        np.complex128_t[:, :, ::1] QEphi_np,
                        np.complex128_t[:, :, ::1] Ephi__np,
                        np.complex128_t[:, :, :, ::1] XQEphi__np,
                        np.complex128_t[:, :, ::1] Ez_np,
                        np.complex128_t[:, :, ::1] QEz_np,
                        np.complex128_t[:, :, ::1] Ezs_np,
                        np.complex128_t[:, :, :, ::1] XQEzs_np, #when called, the list is at the step n-1
                        np.complex128_t[:, :, ::1] Hr_np,
                        np.complex128_t[:, :, ::1] Hrs_np,
                        np.complex128_t[:, :, ::1] QHrs_np,
                        np.complex128_t[:, :, ::1] Hphi_np,
                        np.complex128_t[:, :, ::1] QHphi_np,
                        np.complex128_t[:, :, ::1] Hphi__np,
                        np.complex128_t[:, :, ::1] QHphi__np,
                        np.complex128_t[:, :, :, ::1] XQHphi__np, #when called, the list is at step n-3/2
                        np.complex128_t[:, :, ::1] Hz_np,
                        np.complex128_t[:, :, ::1] QHz_np,
                        np.complex128_t[:, :, ::1] Hzs_np,
                        np.complex128_t[:, :, :, ::1] XQHzs_np,
                        floattype_t[:, :, ::1] alpha,
                        floattype_t[:, :, ::1] sigma,
                        floattype_t[:, :, ::1] kappa,
                        floattype_t[:, :, ::1] b,
                        floattype_t[:, :, ::1] Omega_term_list,
                        floattype_t[:, :, ::1] alpha_term_list,
                        floattype_t[:, :, ::1] Ksi_term_list,
                        floattype_t[:, :, ::1] Lambda_term_list,
                        floattype_t[:, :, ::1] R_term_list,
                        floattype_t[:, :, ::1] Psi_term_list,
                        floattype_t[:, :, ::1] Theta_term_list,

                ):
    """
    
    This function updates all the E fields inside the PML.
    
    Args:
        rs, rf, zs, zf (int): locations of the fields to be updated
        m (int): the argument in e^(i*m*phi) to ensure the symmetry
        dr, dz (float): spatial discretization (no need for dphi as we use the symmetry)
        dt (float): timestep in s
        nz_tot (int): number of cells along the z axis for the whole domain
        nthreads (int): number of threads to use
        alpha, sigma, kappa, b (memoryviews): PML parameters
        Er, Ephi, Ez, Hr, Hphi, Hz (memoryviews): fields in time domain
        Ers, Ephi_, Ezs, Hrs, Hphi_, Hzs (memoryviews): fields used for PML updates

    """
    cdef Py_ssize_t i, k, ii, kk
    cdef int nr
    cdef floattype_t sigma_term, kappa_term, denominateur_kappa_sigma, b_term, R_term, denominateur_R_b, arg
    cdef Py_ssize_t taille_i_fields, taille_j_fields, taille_i_others, taille_j_others, taille_i_X, taille_j_X, taille_k_X, taille_l_X
    cdef double complex QEzs_n

    taille_i_fields = Er_np.shape[0]
    taille_j_fields = Er_np.shape[1]
    taille_i_others = Ers_np.shape[0]
    taille_j_others = Ers_np.shape[1]
    taille_i_X = XQEphi__np.shape[0]
    taille_j_X = XQEphi__np.shape[1]
    taille_k_X = XQEphi__np.shape[2]
    taille_l_X = XQEphi__np.shape[3]

    cdef double complex*** Er = alloc_and_copy_complex3D(Er_np)
    cdef double complex*** Ers = alloc_and_copy_complex3D(Ers_np)
    cdef double complex*** QErs = alloc_and_copy_complex3D(QErs_np)
    cdef double complex*** Ephi = alloc_and_copy_complex3D(Ephi_np)
    cdef double complex*** QEphi = alloc_and_copy_complex3D(QEphi_np)
    cdef double complex*** Ephi_ = alloc_and_copy_complex3D(Ephi__np)
    cdef double complex**** XQEphi_ = alloc_and_copy_complex4D(XQEphi__np)
    cdef double complex*** Ez = alloc_and_copy_complex3D(Ez_np)
    cdef double complex*** QEz = alloc_and_copy_complex3D(QEz_np)
    cdef double complex*** Ezs = alloc_and_copy_complex3D(Ezs_np)
    cdef double complex**** XQEzs = alloc_and_copy_complex4D(XQEzs_np)
    cdef double complex*** Hr = alloc_and_copy_complex3D(Hr_np)
    cdef double complex*** Hrs = alloc_and_copy_complex3D(Hrs_np)
    cdef double complex*** QHrs = alloc_and_copy_complex3D(QHrs_np)
    cdef double complex*** Hphi = alloc_and_copy_complex3D(Hphi_np)
    cdef double complex*** QHphi = alloc_and_copy_complex3D(QHphi_np)
    cdef double complex*** Hphi_ = alloc_and_copy_complex3D(Hphi__np)
    cdef double complex*** QHphi_ = alloc_and_copy_complex3D(QHphi__np)
    cdef double complex**** XQHphi_ = alloc_and_copy_complex4D(XQHphi__np)
    cdef double complex*** Hz = alloc_and_copy_complex3D(Hz_np)
    cdef double complex*** QHz = alloc_and_copy_complex3D(QHz_np)
    cdef double complex*** Hzs = alloc_and_copy_complex3D(Hzs_np)
    cdef double complex**** XQHzs = alloc_and_copy_complex4D(XQHzs_np)


    nr = thickness_r

    for i in prange(0, nr, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + rs
        for k in range(0, nz):

            sigma_term = sigma[i][0][k] / (2*e0)
            kappa_term = kappa[i][0][k] / dt
            denominateur_kappa_sigma = (sigma_term + kappa_term)

            #Er,Qers,Ers-> Utiliser liste Psi
            if k == 0: #Boundary conditions
                Er[ii][0][k] += (1j * m * Hz[ii][0][k] /(dr * (ii-0.5)) - (Hphi[ii][0][k] - 0)/dz)/e0 #OK

            else:
                Er[ii][0][k] += (1j * m * Hz[ii][0][k] /(dr * (ii-0.5)) - (Hphi[ii][0][k] - Hphi[ii][0][k-1])/dz)/e0 #OK
            QErs[i][0][k] = QErs[i][0][k]*alpha_term_list[i][0][k] + Psi_term_list[i][0][k] * Er[ii][0][k] #OK
            Ers[i][0][k] = kappa[i][0][k] * Er[ii][0][k] + QErs[i][0][k] #OK

            #Ephi, QEphi
            if k == 0: #Hrs[ii, 0, kk - 1] = 0 because proportionate to Hr
                Ephi[ii][0][k] = (((kappa_term - sigma_term) * Ephi[ii][0][k] + QEphi[i][0][k] * (
                            1 + alpha_term_list[i][0][k]) + (Hrs[i][0][k] - 0) / (dz * e0) - (
                                    Hz[ii][0][k] - Hz[ii - 1][0][k]) / (dr * e0))
                                / (denominateur_kappa_sigma - Theta_term_list[i][0][k]))  #OK
            else:
                Ephi[ii][0][k] = (((kappa_term - sigma_term) * Ephi[ii][0][k] + QEphi[i][0][k] * (1 + alpha_term_list[i][0][k]) +
                                (Hrs[i][0][k] - Hrs[i][0][k-1])/(dz*e0) - (Hz[ii][0][k] - Hz[ii-1][0][k])/(dr * e0))
                                / (denominateur_kappa_sigma - Theta_term_list[i][0][k])) #OK

            QEphi[i][0][k] = Theta_term_list[i][0][k] * Ephi[ii][0][k] + QEphi[i][0][k] * alpha_term_list[i][0][k]


    # We leave the first for statement to update XQEphi_ and XQEzs
    update_XQEphi_(rs, nz, nthreads, thickness_r, Ephi, XQEphi_, Omega_term_list, alpha_term_list)
    update_XQEzs(rs, nz, nthreads, thickness_r, Ezs, XQEzs, Ksi_term_list, alpha_term_list)


    for i in prange(0, nr, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + rs
        for k in range(0, nz):
            # Ephi_    
            Ephi_[i][0][k] = (b[i][0][k]*Ephi[ii][0][k]  #OK
                            + sum_list_c(XQEphi_, i, 0, k, taille_l_X)) #This sum is in fact QEphi_

            # Ezs
            b_term = b[i][0][k] / dt
            R_term = R_term_list[i][0][k] / 2
            denominateur_R_b = b_term + R_term
            arg = (alpha[i][0][k] * kappa[i][0][k] + sigma[i][0][k]) * dt / (kappa[i][0][k] * e0)
            QEzs_n = sum_list_c(XQEzs, i, 0, k, taille_l_X)


            #I decided to take the derivative of Hphi_ as Hphi_(i+1) - Hphi_(i), which is different from the paper
            # as it is easier to process Hphi_(rmax +  1), which is null because of boundary conditions, than Hphi_(i0-1), which has to be computed
            if i == nr-1:
                Ezs[i][0][k] = ((b_term - R_term) * Ezs[i][0][k] + QEzs_n + simple_sum(Hadamard_product_i(
                    XQEzs, alpha_term_list, i, 0, k, taille_l_X), taille_l_X)  #OK
                                + (0 - Hphi_[i][0][k]) / (dr * e0) -
                                1j * m * Hrs[i][0][k] / e0) / (denominateur_R_b - sum3D_i(Ksi_term_list, 0, k))
            else:
                Ezs[i][0][k] = ((b_term - R_term) * Ezs[i][0][k] + QEzs_n + simple_sum(Hadamard_product_i(XQEzs, alpha_term_list, i, 0, k, taille_l_X), taille_l_X) #OK
                                + (Hphi_[i+1][0][k] - Hphi_[i][0][k])/(dr * e0) -
                                1j * m * Hrs[i][0][k] / e0 )/(denominateur_R_b - sum3D_i(Ksi_term_list, 0, k))

            #QEz
            QEz[i][0][k] = Lambda_term_list[i][0][k] * Ezs[i][0][k] + QEz[i][0][k]*e**(-arg) #OK

            #Ez
            Ez[ii][0][k] = (Ezs[i][0][k] - QEz[i][0][k])/kappa[i][0][k] #OK

    copy_complex3D_to_numpy(Er, Er_np)
    copy_complex3D_to_numpy(Ers, Ers_np)
    copy_complex3D_to_numpy(QErs, QErs_np)
    copy_complex3D_to_numpy(Ephi, Ephi_np)
    copy_complex3D_to_numpy(QEphi, QEphi_np)
    copy_complex3D_to_numpy(Ephi_, Ephi__np)
    copy_complex4D_to_numpy(XQEphi_, XQEphi__np)
    copy_complex3D_to_numpy(Ez, Ez_np)
    copy_complex3D_to_numpy(QEz, QEz_np)
    copy_complex3D_to_numpy(Ezs, Ezs_np)
    copy_complex4D_to_numpy(XQEzs, XQEzs_np)
    copy_complex3D_to_numpy(Hr, Hr_np)
    copy_complex3D_to_numpy(Hrs, Hrs_np)
    copy_complex3D_to_numpy(QHrs, QHrs_np)
    copy_complex3D_to_numpy(Hphi, Hphi_np)
    copy_complex3D_to_numpy(QHphi, QHphi_np)
    copy_complex3D_to_numpy(Hphi_, Hphi__np)
    copy_complex3D_to_numpy(QHphi_, QHphi__np)
    copy_complex4D_to_numpy(XQHphi_, XQHphi__np)
    copy_complex3D_to_numpy(Hz, Hz_np)
    copy_complex3D_to_numpy(QHz, QHz_np)
    copy_complex3D_to_numpy(Hzs, Hzs_np)
    copy_complex4D_to_numpy(XQHzs, XQHzs_np)

    free_complex3D(Er, taille_i_fields, taille_j_fields)
    free_complex3D(Ers, taille_i_others, taille_j_others)
    free_complex3D(QErs, taille_i_others, taille_j_others)
    free_complex3D(Ephi, taille_i_fields, taille_j_fields)
    free_complex3D(QEphi, taille_i_others, taille_j_others)
    free_complex3D(Ephi_, taille_i_others, taille_j_others)
    free_complex4D(XQEphi_, taille_i_X, taille_j_X, taille_k_X)
    free_complex3D(Ez, taille_i_fields, taille_j_fields)
    free_complex3D(QEz, taille_i_others, taille_j_others)
    free_complex3D(Ezs, taille_i_others, taille_j_others)
    free_complex4D(XQEzs, taille_i_X, taille_j_X, taille_k_X)
    free_complex3D(Hr, taille_i_fields, taille_j_fields)
    free_complex3D(Hrs, taille_i_others, taille_j_others)
    free_complex3D(QHrs, taille_i_others, taille_j_others)
    free_complex3D(Hphi, taille_i_fields, taille_j_fields)
    free_complex3D(QHphi, taille_i_others, taille_j_others)
    free_complex3D(Hphi_, taille_i_others, taille_j_others)
    free_complex3D(QHphi_, taille_i_others, taille_j_others)
    free_complex4D(XQHphi_, taille_i_X, taille_j_X, taille_k_X)
    free_complex3D(Hz, taille_i_fields, taille_j_fields)
    free_complex3D(QHz, taille_i_others, taille_j_others)
    free_complex3D(Hzs, taille_i_others, taille_j_others)
    free_complex4D(XQHzs, taille_i_X, taille_j_X, taille_k_X)

cpdef str test(floattype_t[:, :, ::1] rs):
    cdef str test_str = "Ca fonctionne !"
    print(test_str)
    return

cpdef void H_update_r_slab(
                        int rs,
                        int m,
                        int nz,
                        int thickness_r,
                        floattype_t dr,
                        floattype_t dz,
                        floattype_t dt,
                        int nthreads,
                        np.uint32_t[:, :, :, ::1] ID,
                        np.complex128_t[:, :, ::1] Er_np,
                        np.complex128_t[:, :, ::1] Ers_np,
                        np.complex128_t[:, :, ::1] QErs_np,
                        np.complex128_t[:, :, ::1] Ephi_np,
                        np.complex128_t[:, :, ::1] QEphi_np,
                        np.complex128_t[:, :, ::1] Ephi__np,
                        np.complex128_t[:, :, :, ::1] XQEphi__np,
                        np.complex128_t[:, :, ::1] Ez_np,
                        np.complex128_t[:, :, ::1] QEz_np,
                        np.complex128_t[:, :, ::1] Ezs_np,
                        np.complex128_t[:, :, :, ::1] XQEzs_np, #when called, the list is at the step n-1
                        np.complex128_t[:, :, ::1] Hr_np,
                        np.complex128_t[:, :, ::1] Hrs_np,
                        np.complex128_t[:, :, ::1] QHrs_np,
                        np.complex128_t[:, :, ::1] Hphi_np,
                        np.complex128_t[:, :, ::1] QHphi_np,
                        np.complex128_t[:, :, ::1] Hphi__np,
                        np.complex128_t[:, :, ::1] QHphi__np,
                        np.complex128_t[:, :, :, ::1] XQHphi__np, #when called, the list is at step n-3/2
                        np.complex128_t[:, :, ::1] Hz_np,
                        np.complex128_t[:, :, ::1] QHz_np,
                        np.complex128_t[:, :, ::1] Hzs_np,
                        np.complex128_t[:, :, :, ::1] XQHzs_np,
                        floattype_t[:, :, ::1] alpha,
                        floattype_t[:, :, ::1] sigma,
                        floattype_t[:, :, ::1] kappa,
                        floattype_t[:, :, ::1] b,
                        floattype_t[:, :, ::1] Omega_term_list,
                        floattype_t[:, :, ::1] alpha_term_list,
                        floattype_t[:, :, ::1] Ksi_term_list,
                        floattype_t[:, :, ::1] Lambda_term_list,
                        floattype_t[:, :, ::1] R_term_list,
                        floattype_t[:, :, ::1] Psi_term_list,
                        floattype_t[:, :, ::1] Theta_term_list,
                ):
    """

    This function updates all the H fields inside the PML.

    Args:
        rs, rf, zs, zf (int): locations of the fields to be updated
        m (int): the argument in e^(i*m*phi) to ensure the symmetry
        dr, dz (float): spatial discretization (no need for dphi as we use the symmetry)
        dt (float): timestep in s
        nz_tot (int): number of cells along the z axis for the whole domain
        nthreads (int): number of threads to use
        alpha, sigma, kappa, b (memoryviews): PML parameters
        Er, Ephi, Ez, Hr, Hphi, Hz (memoryviews): fields in time domain
        Ers, Ephi_, Ezs, Hrs, Hphi_, Hzs (memoryviews): fields used for PML updates

    """
    cdef Py_ssize_t i, k, ii
    cdef Py_ssize_t taille_i_fields, taille_j_fields, taille_i_others, taille_j_others, taille_i_X, taille_j_X, taille_k_X, taille_l_X
    cdef int nr
    cdef floattype_t sigma_term, kappa_term, denominateur_kappa_sigma, b_term, R_term
    cdef double complex QHzs

    taille_i_fields = Er_np.shape[0]
    taille_j_fields = Er_np.shape[1]
    taille_k_fields = Er_np.shape[2]
    taille_i_others = Ers_np.shape[0]
    taille_j_others = Ers_np.shape[1]
    taille_k_others = Ers_np.shape[2]
    taille_i_X = XQHzs_np.shape[0]
    taille_j_X = XQHzs_np.shape[1]
    taille_k_X = XQHzs_np.shape[2]
    taille_l_X = XQHzs_np.shape[3]

    nr = thickness_r

    cdef double complex*** Er = alloc_and_copy_complex3D(Er_np)
    cdef double complex*** Ers = alloc_and_copy_complex3D(Ers_np)
    cdef double complex*** QErs = alloc_and_copy_complex3D(QErs_np)
    cdef double complex*** Ephi = alloc_and_copy_complex3D(Ephi_np)
    cdef double complex*** QEphi = alloc_and_copy_complex3D(QEphi_np)
    cdef double complex*** Ephi_ = alloc_and_copy_complex3D(Ephi__np)
    cdef double complex**** XQEphi_ = alloc_and_copy_complex4D(XQEphi__np)
    cdef double complex*** Ez = alloc_and_copy_complex3D(Ez_np)
    cdef double complex*** QEz = alloc_and_copy_complex3D(QEz_np)
    cdef double complex*** Ezs = alloc_and_copy_complex3D(Ezs_np)
    cdef double complex**** XQEzs = alloc_and_copy_complex4D(XQEzs_np)
    cdef double complex*** Hr = alloc_and_copy_complex3D(Hr_np)
    cdef double complex*** Hrs = alloc_and_copy_complex3D(Hrs_np)
    cdef double complex*** QHrs = alloc_and_copy_complex3D(QHrs_np)
    cdef double complex*** Hphi = alloc_and_copy_complex3D(Hphi_np)
    cdef double complex*** QHphi = alloc_and_copy_complex3D(QHphi_np)
    cdef double complex*** Hphi_ = alloc_and_copy_complex3D(Hphi__np)
    cdef double complex*** QHphi_ = alloc_and_copy_complex3D(QHphi__np)
    cdef double complex**** XQHphi_ = alloc_and_copy_complex4D(XQHphi__np)
    cdef double complex*** Hz = alloc_and_copy_complex3D(Hz_np)
    cdef double complex*** QHz = alloc_and_copy_complex3D(QHz_np)
    cdef double complex*** Hzs = alloc_and_copy_complex3D(Hzs_np)
    cdef double complex**** XQHzs = alloc_and_copy_complex4D(XQHzs_np)

    for i in prange(0, nr, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + rs
        for k in range(0, nz):
            # Hr, QHrs, Hrs
            if k == 0:
                Hr[ii][0][k] += (Ephi[ii][0][k] - 0) / (dz * mu0) - 1j * m * Ez[ii][0][k] / ((ii - 1) * dr * mu0)  # OK
            else:
                Hr[ii][0][k] += (Ephi[ii][0][k] - Ephi[ii][0][k - 1]) / (dz * mu0) - 1j * m * Ez[ii][0][k] / ((ii - 1) * dr * mu0)  # OK
            QHrs[i][0][k] = Psi_term_list[i][0][k] * Hr[ii][0][k] + QHrs[i][0][k] * alpha_term_list[i][0][k]  # OK
            Hrs[i][0][k] = kappa[i][0][k] * Hr[ii][0][k] + QHrs[i][0][k]  # OK

            # Hphi
            sigma_term = sigma[i][0][k] / (2 * e0)
            kappa_term = kappa[i][0][k] / dt

            if k == nz - 1:  # Ers is proportionate to Er
                if i == nr - 1:
                    Hphi[ii][0][k] = (((kappa_term - sigma_term) * Hphi[ii][0][k] + (1 + alpha_term_list[i][0][k]) *
                                    QHphi[i][0][k]
                                    + (0 - Ez[ii][0][k]) / (dr * mu0) - (
                                            0 - Ers[i][0][k]) / (dz * mu0)) /
                                    (sigma_term + kappa_term - Theta_term_list[i][0][k]))
                else:
                    Hphi[ii][0][k] = (((kappa_term - sigma_term) * Hphi[ii][0][k] + (1 + alpha_term_list[i][0][k]) *
                                    QHphi[i][0][k]
                                    + (Ez[ii + 1][0][k] - Ez[ii][0][k]) / (dr * mu0) - (
                                                0 - Ers[i][0][k]) / (dz * mu0)) /
                                    (sigma_term + kappa_term - Theta_term_list[i][0][k]))
            else:
                if i == nr - 1:
                    Hphi[ii][0][k] = (((kappa_term - sigma_term) * Hphi[ii][0][k] + (1 + alpha_term_list[i][0][k]) *
                                    QHphi[i][0][k]
                                    + (0 - Ez[ii][0][k]) / (dr * mu0) - (
                                                Ers[i][0][k + 1] - Ers[i][0][k]) / (dz * mu0)) /
                                    (sigma_term + kappa_term - Theta_term_list[i][0][k]))
                else:
                    Hphi[ii][0][k] = (((kappa_term - sigma_term) * Hphi[ii][0][k] + (1 + alpha_term_list[i][0][k]) * QHphi[i][0][k]
                                    + (Ez[ii + 1][0][k] - Ez[ii][0][k]) / (dr * mu0) - (Ers[i][0][k + 1] - Ers[i][0][k]) / (dz * mu0)) /
                                    (sigma_term + kappa_term - Theta_term_list[i][0][k]))  # OK

            # QHphi
            QHphi[i][0][k] = Theta_term_list[i][0][k] * Hphi[ii][0][k] + QHphi[i][0][k] * alpha_term_list[i][0][k]


    # We leave the for statement to update XQHphi_ and XQHzs
    update_XQHphi_(rs, nz, nthreads, thickness_r, Hphi, XQHphi_, Omega_term_list, alpha_term_list)
    update_XQHzs(rs, nz, nthreads, thickness_r, Hzs, XQHzs, Ksi_term_list, alpha_term_list)


    for i in prange(0, nr, nogil=True, schedule='static', num_threads=nthreads):
        ii = i + rs
        for k in range(0, nz):
            # Hphi_
            Hphi_[i][0][k] = (b[i][0][k] * Hphi[ii][0][k]  # OK
                            + sum4D_i(XQHphi_, i, 0, k, taille_l_X))  # QHphi_

            # Hzs
            b_term = b[i][0][k] / dt
            R_term = R_term_list[i][0][k] / 2
            QHzs = sum_list_c(XQHzs, i, 0, k, taille_l_X)

            if i == nr - 1:  # Ephi_ is proportionate to Ephi at this point (at different times)
                Hzs[i][0][k] = ((b_term - R_term) * Hzs[i][0][k] + (
                        simple_sum(Hadamard_product_i(XQHzs, alpha_term_list, i, 0, k, taille_l_X), taille_l_X) + QHzs) / mu0
                                + Ers[i][0][k] * 1j * m / e0 - (0 - Ephi_[i][0][k]) / (dr * e0)) / (
                                            b_term + R_term
                                            - sum3D_i(Ksi_term_list, 0, k) / mu0)
            else:
                Hzs[i][0][k] = ((b_term - R_term) * Hzs[i][0][k] + (simple_sum(Hadamard_product_i(XQHzs, alpha_term_list, i, 0, k, taille_l_X), taille_l_X) + QHzs) / mu0
                                + Ers[i][0][k] * 1j * m / e0 - (Ephi_[i + 1][0][k] - Ephi_[i][0][k]) / (dr * e0)) / (b_term + R_term - sum3D_i(Ksi_term_list, 0, k) / mu0)  # OK

            # QHz
            QHz[i][0][k] = (Lambda_term_list[i][0][k] * Hzs[i][0][k] +
                            QHz[i][0][k] * e**(-(alpha[i][0][k] * kappa[i][0][k] + sigma[i][0][k]) * dt / (kappa[i][0][k] * e0)))

            # Hz
            Hz[ii][0][k] = (Hzs[i][0][k] - QHz[i][0][k]) / kappa[i][0][k]

    copy_complex3D_to_numpy(Er, Er_np)
    copy_complex3D_to_numpy(Ers, Ers_np)
    copy_complex3D_to_numpy(QErs, QErs_np)
    copy_complex3D_to_numpy(Ephi, Ephi_np)
    copy_complex3D_to_numpy(QEphi, QEphi_np)
    copy_complex3D_to_numpy(Ephi_, Ephi__np)
    copy_complex4D_to_numpy(XQEphi_, XQEphi__np)
    copy_complex3D_to_numpy(Ez, Ez_np)
    copy_complex3D_to_numpy(QEz, QEz_np)
    copy_complex3D_to_numpy(Ezs, Ezs_np)
    copy_complex4D_to_numpy(XQEzs, XQEzs_np)
    copy_complex3D_to_numpy(Hr, Hr_np)
    copy_complex3D_to_numpy(Hrs, Hrs_np)
    copy_complex3D_to_numpy(QHrs, QHrs_np)
    copy_complex3D_to_numpy(Hphi, Hphi_np)
    copy_complex3D_to_numpy(QHphi, QHphi_np)
    copy_complex3D_to_numpy(Hphi_, Hphi__np)
    copy_complex3D_to_numpy(QHphi_, QHphi__np)
    copy_complex4D_to_numpy(XQHphi_, XQHphi__np)
    copy_complex3D_to_numpy(Hz, Hz_np)
    copy_complex3D_to_numpy(QHz, QHz_np)
    copy_complex3D_to_numpy(Hzs, Hzs_np)
    copy_complex4D_to_numpy(XQHzs, XQHzs_np)

    free_complex3D(Er, taille_i_fields, taille_j_fields)
    free_complex3D(Ers, taille_i_others, taille_j_others)
    free_complex3D(QErs, taille_i_others, taille_j_others)
    free_complex3D(Ephi, taille_i_fields, taille_j_fields)
    free_complex3D(QEphi, taille_i_others, taille_j_others)
    free_complex3D(Ephi_, taille_i_others, taille_j_others)
    free_complex4D(XQEphi_, taille_i_X, taille_j_X, taille_k_X)
    free_complex3D(Ez, taille_i_fields, taille_j_fields)
    free_complex3D(QEz, taille_i_others, taille_j_others)
    free_complex3D(Ezs, taille_i_others, taille_j_others)
    free_complex4D(XQEzs, taille_i_X, taille_j_X, taille_k_X)
    free_complex3D(Hr, taille_i_fields, taille_j_fields)
    free_complex3D(Hrs, taille_i_others, taille_j_others)
    free_complex3D(QHrs, taille_i_others, taille_j_others)
    free_complex3D(Hphi, taille_i_fields, taille_j_fields)
    free_complex3D(QHphi, taille_i_others, taille_j_others)
    free_complex3D(Hphi_, taille_i_others, taille_j_others)
    free_complex3D(QHphi_, taille_i_others, taille_j_others)
    free_complex4D(XQHphi_, taille_i_X, taille_j_X, taille_k_X)
    free_complex3D(Hz, taille_i_fields, taille_j_fields)
    free_complex3D(QHz, taille_i_others, taille_j_others)
    free_complex3D(Hzs, taille_i_others, taille_j_others)
    free_complex4D(XQHzs, taille_i_X, taille_j_X, taille_k_X)


########################################################################################

################## Update of the PMLs in the z direction ###############################

#For this part, it appears that the PML formulation for the z component is in fact the same as in cartesian

cpdef void initialize_constant_lists_z(
        int rs,
        int thickness_z,
        floattype_t dt,
        int nthreads,
        floattype_t[:, :, ::1] alpha_z,
        floattype_t[:, :, ::1] sigma_z,
        floattype_t[:, :, ::1] kappa_z,
        floattype_t[:, :, ::1] Pi_term_list,
        floattype_t[:, :, ::1] Delta_term_list,
        floattype_t[:, :, ::1] Rho_term_list
):
    cdef Py_ssize_t k,i
    cdef floattype_t arg
    for k in prange(0, 2*thickness_z, nogil= True, schedule='static', num_threads = nthreads):
        for i in range(0,rs):
            arg = alpha_z[i,0,k] * kappa_z[i,0,k] + sigma_z[i,0,k]
            Pi_term_list[i,0,k] = (alpha_z[i,0,k] - arg)/(e0 * kappa_z[i,0,k])
            Delta_term_list[i,0,k] = (1 - kappa_z[i,0,k])/kappa_z[i,0,k]
            Rho_term_list[i,0,k] = arg / (e0 * kappa_z[i,0,k])

cpdef void E_update_upper_slab(
        int rs, #The PML will go from r=0 to r=rs
        int zs,
        int thickness,
        floattype_t dr,
        floattype_t dz,
        floattype_t dt,
        int m,
        int nthreads,
        np.complex128_t[:, :, ::1] Er_np,
        np.complex128_t[:, :, ::1] Ephi_np,
        np.complex128_t[:, :, ::1] Ez_np,
        np.complex128_t[:, :, ::1] Hr_np,
        np.complex128_t[:, :, ::1] Hphi_np,
        np.complex128_t[:, :, ::1] Hz_np,
        np.complex128_t[:, :, ::1] JEphi_np,
        np.complex128_t[:, :, ::1] JEr_np,
        np.complex128_t[:, :, ::1] QEphi_np,
        np.complex128_t[:, :, ::1] QEr_np,
        np.complex128_t[:, :, ::1] QJEphi_np,
        np.complex128_t[:, :, ::1] QJEr_np,
        floattype_t[:, :, ::1] Pi_term_list,
        floattype_t[:, :, ::1] Delta_term_list,
        floattype_t[:, :, ::1] Rho_term_list
):

    cdef Py_ssize_t i,k,kk
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t taille_i_fields, taille_j_fields, taille_i_others, taille_j_others
    cdef int nz, nr

    taille_i_fields = Er_np.shape[0]
    taille_j_fields = Er_np.shape[1]
    taille_i_others = JEr_np.shape[0]
    taille_j_others = JEr_np.shape[1]

    cdef double complex*** Er = alloc_and_copy_complex3D(Er_np)
    cdef double complex*** Ephi = alloc_and_copy_complex3D(Ephi_np)
    cdef double complex*** Ez = alloc_and_copy_complex3D(Ez_np)
    cdef double complex*** Hr = alloc_and_copy_complex3D(Hr_np)
    cdef double complex*** Hphi = alloc_and_copy_complex3D(Hphi_np)
    cdef double complex*** Hz = alloc_and_copy_complex3D(Hz_np)
    cdef double complex*** JEphi = alloc_and_copy_complex3D(JEphi_np)
    cdef double complex*** JEr = alloc_and_copy_complex3D(JEr_np)
    cdef double complex*** QEphi = alloc_and_copy_complex3D(QEphi_np)
    cdef double complex*** QEr = alloc_and_copy_complex3D(QEr_np)
    cdef double complex*** QJEphi = alloc_and_copy_complex3D(QJEphi_np)
    cdef double complex*** QJEr = alloc_and_copy_complex3D(QJEr_np)

    nz = thickness
    nr = rs

    for i in prange(0, nr, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            kk = k + zs

            # Updating the Q lists before updating the fields
            QEr[i][j][k + nz] += (Er[i][j][kk + 1] - Er[i][j][kk]) * dt / dz
            QEphi[i][j][k + nz] += (Ephi[i][j][kk + 1] - Ephi[i][j][kk]) * dt / dz
            QJEr[i][j][k + nz] += JEr[i][j][k + nz] * dt
            QJEphi[i][j][k + nz] += JEphi[i][j][k + nz] * dt

            # Updating the E-fields
            Er[i][j][kk] += (I * m * Hz[i][j][kk] / ((i + 0.5) * dr)
                            - (Hphi[i][j][kk + 1] - Hphi[i][j][kk]) / dz
                            - JEphi[i][j][k]) / e0

            Ephi[i][j][kk] += ((Hr[i][j][kk + 1] - Hr[i][j][kk]) / dz
                            - (Hz[i + 1][j][kk] - Hz[i][j][kk]) / dr
                            + JEphi[i][j][k]) / e0

            Ez[i][j][kk] += (((i + 1.5) * Hphi[i + 1][j][kk] - (i + 0.5) * Hphi[i][j][kk]) / ((i + 0.5) * dr)
                            - I * m * Hr[i][j][kk] / ((i + 0.5) * dr)
                            + JEphi[i][j][k]) / e0

            # Updating the J fields
            JEr[i][j][k] = (Pi_term_list[i, j, k + nz] * QEr[i][j][k]
                            + Delta_term_list[i, j, k + nz] * (Er[i][j][kk + 1] - Er[i][j][kk]) / dz
                            - Rho_term_list[i, j, k + nz] * QJEr[i][j][k])

            JEphi[i][j][k] = (Pi_term_list[i, j, k + nz] * QEphi[i][j][k]
                            + Delta_term_list[i, j, k + nz] * (Ephi[i][j][kk + 1] - Ephi[i][j][kk]) / dz
                            - Rho_term_list[i, j, k + nz] * QJEphi[i][j][k])

    copy_complex3D_to_numpy(Er, Er_np)
    copy_complex3D_to_numpy(Ephi, Ephi_np)
    copy_complex3D_to_numpy(Ez, Ez_np)
    copy_complex3D_to_numpy(Hr, Hr_np)
    copy_complex3D_to_numpy(Hphi, Hphi_np)
    copy_complex3D_to_numpy(Hz, Hz_np)
    copy_complex3D_to_numpy(JEphi, JEphi_np)
    copy_complex3D_to_numpy(JEr, JEr_np)
    copy_complex3D_to_numpy(QEphi, QEphi_np)
    copy_complex3D_to_numpy(QEr, QEr_np)
    copy_complex3D_to_numpy(QJEphi, QJEphi_np)
    copy_complex3D_to_numpy(QJEr, QJEr_np)

    free_complex3D(Er, taille_i_fields, taille_j_fields)
    free_complex3D(Ephi, taille_i_fields, taille_j_fields)
    free_complex3D(Ez, taille_i_fields, taille_j_fields)
    free_complex3D(Hr, taille_i_fields, taille_j_fields)
    free_complex3D(Hphi, taille_i_fields, taille_j_fields)
    free_complex3D(Hz, taille_i_fields, taille_j_fields)
    free_complex3D(JEphi, taille_i_others, taille_j_others)
    free_complex3D(JEr, taille_i_others, taille_j_others)
    free_complex3D(QEphi, taille_i_others, taille_j_others)
    free_complex3D(QEr, taille_i_others, taille_j_others)
    free_complex3D(QJEphi, taille_i_others, taille_j_others)
    free_complex3D(QJEr, taille_i_others, taille_j_others)

cpdef void H_update_upper_slab(
        int rs, #The PML will go from r=0 to r=rs
        int zs,
        int thickness,
        floattype_t dr,
        floattype_t dz,
        floattype_t dt,
        int m,
        int nthreads,
        np.complex128_t[:, :, ::1] Er_np,
        np.complex128_t[:, :, ::1] Ephi_np,
        np.complex128_t[:, :, ::1] Ez_np,
        np.complex128_t[:, :, ::1] Hr_np,
        np.complex128_t[:, :, ::1] Hphi_np,
        np.complex128_t[:, :, ::1] Hz_np,
        np.complex128_t[:, :, ::1] JHphi_np,
        np.complex128_t[:, :, ::1] JHr_np,
        np.complex128_t[:, :, ::1] QHphi_np,
        np.complex128_t[:, :, ::1] QHr_np,
        np.complex128_t[:, :, ::1] QJHphi_np,
        np.complex128_t[:, :, ::1] QJHr_np,
        floattype_t[:, :, ::1] Pi_term_list, #Pi_term_list[i,0,k] gives the Pi value at (i,0,k) if k <= nz-1, else at (i,0,nz-thickness+k)
        floattype_t[:, :, ::1] Delta_term_list,
        floattype_t[:, :, ::1] Rho_term_list
):

    cdef Py_ssize_t i, k, kk
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t taille_i_fields, taille_j_fields, taille_i_others, taille_j_others
    cdef int nz, nr

    taille_i_fields = Er_np.shape[0]
    taille_j_fields = Er_np.shape[1]
    taille_i_others = JHr_np.shape[0]
    taille_j_others = JHr_np.shape[1]


    cdef double complex*** Er = alloc_and_copy_complex3D(Er_np)
    cdef double complex*** Ephi = alloc_and_copy_complex3D(Ephi_np)
    cdef double complex*** Ez = alloc_and_copy_complex3D(Ez_np)
    cdef double complex*** Hr = alloc_and_copy_complex3D(Hr_np)
    cdef double complex*** Hphi = alloc_and_copy_complex3D(Hphi_np)
    cdef double complex*** Hz = alloc_and_copy_complex3D(Hz_np)
    cdef double complex*** JHphi = alloc_and_copy_complex3D(JHphi_np)
    cdef double complex*** JHr = alloc_and_copy_complex3D(JHr_np)
    cdef double complex*** QHphi = alloc_and_copy_complex3D(QHphi_np)
    cdef double complex*** QHr = alloc_and_copy_complex3D(QHr_np)
    cdef double complex*** QJHphi = alloc_and_copy_complex3D(QJHphi_np)
    cdef double complex*** QJHr = alloc_and_copy_complex3D(QJHr_np)

    nz = thickness
    nr = rs

    for i in prange(0, nr, nogil=True, schedule='static', num_threads= nthreads):
        for k in range(0, nz):
            kk = k + zs

            # Updating the Q lists before updating the fields
            QHr[i][j][k + nz] += (Hr[i][j][kk + 1] - Hr[i][j][kk]) * dt / dz
            QHphi[i][j][k + nz] += (Hphi[i][j][kk + 1] - Hphi[i][j][kk]) * dt / dz
            QJHr[i][j][k + nz] += JHr[i][j][k + nz] * dt
            QJHphi[i][j][k + nz] += JHphi[i][j][k + nz] * dt

            # Updating the E-fields
            Hr[i][j][kk] -= (I * m * Ez[i][j][kk] / ((i + 1) * dr)
                            - (Ephi[i][j][kk + 1] - Ephi[i][j][kk]) / dz
                            - JHphi[i][j][k]) / mu0

            Hphi[i][j][kk] -= ((Er[i][j][kk + 1] - Er[i][j][kk]) / dz
                            - (Ez[i + 1][j][kk] - Ez[i][j][kk]) / dr
                            + JHphi[i][j][k]) / mu0

            Hz[i][j][kk] -= (((i + 2) * Ephi[i + 1][j][kk] - (i + 1) * Ephi[i][j][kk]) / ((i + 1) * dr)
                            - I * m * Er[i][j][kk] / ((i + 0.5) * dr)
                            + JHphi[i][j][k]) / mu0
                

            # Updating the J fields
            JHr[i][j][k + nz] = (Pi_term_list[i, j, k + nz] * QHr[i][j][k + nz]
                                + Delta_term_list[i, j, k + nz] * (Hr[i][j][kk + 1] - Hr[i][j][kk]) / dz
                                - Rho_term_list[i, j, k + nz] * QJHr[i][j][k + nz])


            JHphi[i][j][k + nz] = (Pi_term_list[i, j, k + nz] * QHphi[i][j][k + nz]
                                + Delta_term_list[i, j, k + nz] * (Hphi[i][j][kk + 1] - Hphi[i][j][kk]) / dz
                                - Rho_term_list[i, j, k + nz] * QJHphi[i][j][k + nz])
    
    copy_complex3D_to_numpy(Er, Er_np)
    copy_complex3D_to_numpy(Ephi, Ephi_np)
    copy_complex3D_to_numpy(Ez, Ez_np)
    copy_complex3D_to_numpy(Hr, Hr_np)
    copy_complex3D_to_numpy(Hphi, Hphi_np)
    copy_complex3D_to_numpy(Hz, Hz_np)
    copy_complex3D_to_numpy(JHphi, JHphi_np)
    copy_complex3D_to_numpy(JHr, JHr_np)
    copy_complex3D_to_numpy(QHphi, QHphi_np)
    copy_complex3D_to_numpy(QHr, QHr_np)
    copy_complex3D_to_numpy(QJHphi, QJHphi_np)
    copy_complex3D_to_numpy(QJHr, QJHr_np)

    free_complex3D(Er, taille_i_fields, taille_j_fields)
    free_complex3D(Ephi, taille_i_fields, taille_j_fields)
    free_complex3D(Ez, taille_i_fields, taille_j_fields)
    free_complex3D(Hr, taille_i_fields, taille_j_fields)
    free_complex3D(Hphi, taille_i_fields, taille_j_fields)
    free_complex3D(Hz, taille_i_fields, taille_j_fields)
    free_complex3D(JHphi, taille_i_others, taille_j_others)
    free_complex3D(JHr, taille_i_others, taille_j_others)
    free_complex3D(QHphi, taille_i_others, taille_j_others)
    free_complex3D(QHr, taille_i_others, taille_j_others)
    free_complex3D(QJHphi, taille_i_others, taille_j_others)
    free_complex3D(QJHr, taille_i_others, taille_j_others)
    

#For the lower slab, we just need to invert the direction of scanning z, as well as updating the indexes of term_lists

cpdef void E_update_lower_slab(
        int rs, #The PML will go from r=0 to r=rs
        int zf,
        int thickness,
        floattype_t dr,
        floattype_t dz,
        floattype_t dt,
        int m,
        int nthreads,
        np.complex128_t[:, :, ::1] Er_np,
        np.complex128_t[:, :, ::1] Ephi_np,
        np.complex128_t[:, :, ::1] Ez_np,
        np.complex128_t[:, :, ::1] Hr_np,
        np.complex128_t[:, :, ::1] Hphi_np,
        np.complex128_t[:, :, ::1] Hz_np,
        np.complex128_t[:, :, ::1] JEphi_np,
        np.complex128_t[:, :, ::1] JEr_np,
        np.complex128_t[:, :, ::1] QEphi_np,
        np.complex128_t[:, :, ::1] QEr_np,
        np.complex128_t[:, :, ::1] QJEphi_np,
        np.complex128_t[:, :, ::1] QJEr_np,
        floattype_t[:, :, ::1] Pi_term_list,
        floattype_t[:, :, ::1] Delta_term_list,
        floattype_t[:, :, ::1] Rho_term_list
):

    cdef Py_ssize_t i,k,kk
    cdef Py_ssize_t j = 0
    cdef Py_ssize_t taille_i_fields, taille_j_fields, taille_i_others, taille_j_others
    cdef int nz, nr

    taille_i_fields = Er_np.shape[0]
    taille_j_fields = Er_np.shape[1]
    taille_i_others = JEr_np.shape[0]
    taille_j_others = JEr_np.shape[1]

    cdef double complex*** Er = alloc_and_copy_complex3D(Er_np)
    cdef double complex*** Ephi = alloc_and_copy_complex3D(Ephi_np)
    cdef double complex*** Ez = alloc_and_copy_complex3D(Ez_np)
    cdef double complex*** Hr = alloc_and_copy_complex3D(Hr_np)
    cdef double complex*** Hphi = alloc_and_copy_complex3D(Hphi_np)
    cdef double complex*** Hz = alloc_and_copy_complex3D(Hz_np)
    cdef double complex*** JEphi = alloc_and_copy_complex3D(JEphi_np)
    cdef double complex*** JEr = alloc_and_copy_complex3D(JEr_np)
    cdef double complex*** QEphi = alloc_and_copy_complex3D(QEphi_np)
    cdef double complex*** QEr = alloc_and_copy_complex3D(QEr_np)
    cdef double complex*** QJEphi = alloc_and_copy_complex3D(QJEphi_np)
    cdef double complex*** QJEr = alloc_and_copy_complex3D(QJEr_np)


    nz = thickness
    nr = rs

    for i in prange(0, nr, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            kk = nz - (k + 1)  # The first PML cell is at kk = nz-1

            # Updating the Q lists before updating the fields
            QEr[i][j][k] += (Er[i][j][kk + 1] - Er[i][j][kk]) * dt / dz
            QEphi[i][j][k] += (Ephi[i][j][kk + 1] - Ephi[i][j][kk]) * dt / dz
            QJEr[i][j][k] += JEr[i][j][k] * dt
            QJEphi[i][j][k] += JEphi[i][j][k] * dt

            # Updating the E-fields
            Er[i][j][kk] += (I * m * Hz[i][j][kk] / ((i + 0.5) * dr)
                            - (Hphi[i][j][kk + 1] - Hphi[i][j][kk]) / dz
                            - JEphi[i][j][k]) / e0

            Ephi[i][j][kk] += ((Hr[i][j][kk + 1] - Hr[i][j][kk]) / dz
                            - (Hz[i + 1][j][kk] - Hz[i][j][kk]) / dr
                            + JEphi[i][j][k]) / e0

            Ez[i][j][kk] += (((i + 1.5) * Hphi[i + 1][j][kk] - (i + 0.5) * Hphi[i][j][kk])
                            / ((i + 0.5) * dr)
                            - I * m * Hr[i][j][kk] / ((i + 0.5) * dr)
                            + JEphi[i][j][k]) / e0

            # Updating the J fields
            JEr[i][j][k] = (Pi_term_list[i, j, k] * QEr[i][j][k]
                            + Delta_term_list[i, j, k] * (Er[i][j][kk + 1] - Er[i][j][kk]) / dz
                            - Rho_term_list[i, j, k] * QJEr[i][j][k])

            JEphi[i][j][k] = (Pi_term_list[i, j, k] * QEphi[i][j][k]
                            + Delta_term_list[i, j, k] * (Ephi[i][j][kk + 1] - Ephi[i][j][kk]) / dz
                            - Rho_term_list[i, j, k] * QJEphi[i][j][k])
    
    copy_complex3D_to_numpy(Er, Er_np)
    copy_complex3D_to_numpy(Ephi, Ephi_np)
    copy_complex3D_to_numpy(Ez, Ez_np)
    copy_complex3D_to_numpy(Hr, Hr_np)
    copy_complex3D_to_numpy(Hphi, Hphi_np)
    copy_complex3D_to_numpy(Hz, Hz_np)
    copy_complex3D_to_numpy(JEphi, JEphi_np)
    copy_complex3D_to_numpy(JEr, JEr_np)
    copy_complex3D_to_numpy(QEphi, QEphi_np)
    copy_complex3D_to_numpy(QEr, QEr_np)
    copy_complex3D_to_numpy(QJEphi, QJEphi_np)
    copy_complex3D_to_numpy(QJEr, QJEr_np)

    free_complex3D(Er, taille_i_fields, taille_j_fields)
    free_complex3D(Ephi, taille_i_fields, taille_j_fields)
    free_complex3D(Ez, taille_i_fields, taille_j_fields)
    free_complex3D(Hr, taille_i_fields, taille_j_fields)
    free_complex3D(Hphi, taille_i_fields, taille_j_fields)
    free_complex3D(Hz, taille_i_fields, taille_j_fields)
    free_complex3D(JEphi, taille_i_others, taille_j_others)
    free_complex3D(JEr, taille_i_others, taille_j_others)
    free_complex3D(QEphi, taille_i_others, taille_j_others)
    free_complex3D(QEr, taille_i_others, taille_j_others)
    free_complex3D(QJEphi, taille_i_others, taille_j_others)
    free_complex3D(QJEr, taille_i_others, taille_j_others)
    

cpdef void H_update_lower_slab(
        int rs, #The PML will go from r=0 to r=rs
        int zf,
        int thickness,
        floattype_t dr,
        floattype_t dz,
        floattype_t dt,
        int m,
        int nthreads,
        np.complex128_t[:, :, ::1] Er_np,
        np.complex128_t[:, :, ::1] Ephi_np,
        np.complex128_t[:, :, ::1] Ez_np,
        np.complex128_t[:, :, ::1] Hr_np,
        np.complex128_t[:, :, ::1] Hphi_np,
        np.complex128_t[:, :, ::1] Hz_np,
        np.complex128_t[:, :, ::1] JHphi_np,
        np.complex128_t[:, :, ::1] JHr_np,
        np.complex128_t[:, :, ::1] QHphi_np,
        np.complex128_t[:, :, ::1] QHr_np,
        np.complex128_t[:, :, ::1] QJHphi_np,
        np.complex128_t[:, :, ::1] QJHr_np,
        floattype_t[:, :, ::1] Pi_term_list, #Pi_term_list[i,0,k] gives the Pi value at (i,0,k) if k <= nz-1, else at (i,0,nz-thickness+k)
        floattype_t[:, :, ::1] Delta_term_list,
        floattype_t[:, :, ::1] Rho_term_list
):

    cdef Py_ssize_t i, k, kk
    cdef Py_ssize_t taille_i_fields, taille_j_fields, taille_i_others, taille_j_others
    cdef int nz, nr

    taille_i_fields = Er_np.shape[0]
    taille_j_fields = Er_np.shape[1]
    taille_i_others = JHr_np.shape[0]
    taille_j_others = JHr_np.shape[1]


    cdef double complex*** Er = alloc_and_copy_complex3D(Er_np)
    cdef double complex*** Ephi = alloc_and_copy_complex3D(Ephi_np)
    cdef double complex*** Ez = alloc_and_copy_complex3D(Ez_np)
    cdef double complex*** Hr = alloc_and_copy_complex3D(Hr_np)
    cdef double complex*** Hphi = alloc_and_copy_complex3D(Hphi_np)
    cdef double complex*** Hz = alloc_and_copy_complex3D(Hz_np)
    cdef double complex*** JHphi = alloc_and_copy_complex3D(JHphi_np)
    cdef double complex*** JHr = alloc_and_copy_complex3D(JHr_np)
    cdef double complex*** QHphi = alloc_and_copy_complex3D(QHphi_np)
    cdef double complex*** QHr = alloc_and_copy_complex3D(QHr_np)
    cdef double complex*** QJHphi = alloc_and_copy_complex3D(QJHphi_np)
    cdef double complex*** QJHr = alloc_and_copy_complex3D(QJHr_np)

    nz = thickness
    nr = rs

    for i in prange(0, nr, nogil=True, schedule='static', num_threads=nthreads):
        for k in range(0, nz):
            kk = nz - (k + 1)  # The first PML cell is at kk = nz-1
            
            # Updating the Q lists before updating the fields
            QHr[i][0][k] += (Hr[i][0][kk + 1] - Hr[i][0][kk]) * dt / dz
            QHphi[i][0][k] += (Hphi[i][0][kk + 1] - Hphi[i][0][kk]) * dt / dz
            QJHr[i][0][k] += JHr[i][0][k] * dt
            QJHphi[i][0][k] += JHphi[i][0][k] * dt

            # Updating the E-fields
            Hr[i][0][kk] -= (I * m * Ez[i][0][kk] / ((i + 1) * dr) - (Ephi[i][0][kk + 1] - Ephi[i][0][kk]) / dz -
                            JHphi[i][0][k]) / mu0
            Hphi[i][0][kk] -= ((Er[i][0][kk + 1] - Er[i][0][kk]) / dz - (Ez[i + 1][0][kk] - Ez[i][0][kk]) / dr +
                            JHphi[i][0][k]) / mu0
            Hz[i][0][kk] -= (((i + 2) * Ephi[i + 1][0][kk] - (i + 1) * Ephi[i][0][kk]) / ((i + 1) * dr) - 
                            I * m * Er[i][0][kk] / ((i + 0.5) * dr) + JHphi[i][0][k]) / mu0

            # Updating the J fields
            JHr[i][0][k] = Pi_term_list[i, 0, k] * QHr[i][0][k] + Delta_term_list[i, 0, k] * (
                            Hr[i][0][kk + 1] - Hr[i][0][kk]) / dz - Rho_term_list[i, 0, k] * QJHr[i][0][k]
            JHphi[i][0][k] = Pi_term_list[i, 0, k] * QHphi[i][0][k] + Delta_term_list[i, 0, k] * (
                            Hphi[i][0][kk + 1] - Hphi[i][0][kk]) / dz - Rho_term_list[i, 0, k] * QJHphi[i][0][k]
    
    copy_complex3D_to_numpy(Er, Er_np)
    copy_complex3D_to_numpy(Ephi, Ephi_np)
    copy_complex3D_to_numpy(Ez, Ez_np)
    copy_complex3D_to_numpy(Hr, Hr_np)
    copy_complex3D_to_numpy(Hphi, Hphi_np)
    copy_complex3D_to_numpy(Hz, Hz_np)
    copy_complex3D_to_numpy(JHphi, JHphi_np)
    copy_complex3D_to_numpy(JHr, JHr_np)
    copy_complex3D_to_numpy(QHphi, QHphi_np)
    copy_complex3D_to_numpy(QHr, QHr_np)
    copy_complex3D_to_numpy(QJHphi, QJHphi_np)
    copy_complex3D_to_numpy(QJHr, QJHr_np)

    free_complex3D(Er, taille_i_fields, taille_j_fields)
    free_complex3D(Ephi, taille_i_fields, taille_j_fields)
    free_complex3D(Ez, taille_i_fields, taille_j_fields)
    free_complex3D(Hr, taille_i_fields, taille_j_fields)
    free_complex3D(Hphi, taille_i_fields, taille_j_fields)
    free_complex3D(Hz, taille_i_fields, taille_j_fields)
    free_complex3D(JHphi, taille_i_others, taille_j_others)
    free_complex3D(JHr, taille_i_others, taille_j_others)
    free_complex3D(QHphi, taille_i_others, taille_j_others)
    free_complex3D(QHr, taille_i_others, taille_j_others)
    free_complex3D(QJHphi, taille_i_others, taille_j_others)
    free_complex3D(QJHr, taille_i_others, taille_j_others)
    