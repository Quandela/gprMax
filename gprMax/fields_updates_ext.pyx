# Copyright (C) 2015-2023: The University of Edinburgh
#                 Authors: Craig Warren and Antonis Giannopoulos
#
# This file is part of gprMax.
#
# gprMax is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gprMax is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gprMax.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
cimport numpy as np
from cython.parallel import prange

from gprMax.constants cimport floattype_t
from gprMax.constants cimport complextype_t
from scipy.constants import epsilon_0 as e0
from scipy.constants import mu_0 as mu0


###############################################
# Electric field updates - standard materials #
###############################################
cpdef void update_electric(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    floattype_t[:, ::1] updatecoeffsE,
                    np.uint32_t[:, :, :, ::1] ID,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez,
                    floattype_t[:, :, ::1] Hx,
                    floattype_t[:, :, ::1] Hy,
                    floattype_t[:, :, ::1] Hz
            ):
    """This function updates the electric field components.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """

    cdef Py_ssize_t i, j, k
    cdef int materialEx, materialEy, materialEz

    # 2D - Ex component
    if nx == 1:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    materialEx = ID[0, i, j, k]
                    Ex[i, j, k] = updatecoeffsE[materialEx, 0] * Ex[i, j, k] + updatecoeffsE[materialEx, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[materialEx, 3] * (Hy[i, j, k] - Hy[i, j, k - 1])

    # 2D - Ey component
    elif ny == 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    materialEy = ID[1, i, j, k]
                    Ey[i, j, k] = updatecoeffsE[materialEy, 0] * Ey[i, j, k] + updatecoeffsE[materialEy, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[materialEy, 1] * (Hz[i, j, k] - Hz[i - 1, j, k])

    # 2D - Ez component
    elif nz == 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    materialEz = ID[2, i, j, k]
                    Ez[i, j, k] = updatecoeffsE[materialEz, 0] * Ez[i, j, k] + updatecoeffsE[materialEz, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[materialEz, 2] * (Hx[i, j, k] - Hx[i, j - 1, k])

    # 3D
    else:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    materialEx = ID[0, i, j, k]
                    materialEy = ID[1, i, j, k]
                    materialEz = ID[2, i, j, k]
                    Ex[i, j, k] = updatecoeffsE[materialEx, 0] * Ex[i, j, k] + updatecoeffsE[materialEx, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[materialEx, 3] * (Hy[i, j, k] - Hy[i, j, k - 1])
                    Ey[i, j, k] = updatecoeffsE[materialEy, 0] * Ey[i, j, k] + updatecoeffsE[materialEy, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[materialEy, 1] * (Hz[i, j, k] - Hz[i - 1, j, k])
                    Ez[i, j, k] = updatecoeffsE[materialEz, 0] * Ez[i, j, k] + updatecoeffsE[materialEz, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[materialEz, 2] * (Hx[i, j, k] - Hx[i, j - 1, k])

        # Ex components at i = 0
        for j in prange(1, ny, nogil=True, schedule='static', num_threads=nthreads):
            for k in range(1, nz):
                materialEx = ID[0, 0, j, k]
                Ex[0, j, k] = updatecoeffsE[materialEx, 0] * Ex[0, j, k] + updatecoeffsE[materialEx, 2] * (Hz[0, j, k] - Hz[0, j - 1, k]) - updatecoeffsE[materialEx, 3] * (Hy[0, j, k] - Hy[0, j, k - 1])

        # Ey components at j = 0
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for k in range(1, nz):
                materialEy = ID[1, i, 0, k]
                Ey[i, 0, k] = updatecoeffsE[materialEy, 0] * Ey[i, 0, k] + updatecoeffsE[materialEy, 3] * (Hx[i, 0, k] - Hx[i, 0, k - 1]) - updatecoeffsE[materialEy, 1] * (Hz[i, 0, k] - Hz[i - 1, 0, k])

        # Ez components at k = 0
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                materialEz = ID[2, i, j, 0]
                Ez[i, j, 0] = updatecoeffsE[materialEz, 0] * Ez[i, j, 0] + updatecoeffsE[materialEz, 1] * (Hy[i, j, 0] - Hy[i - 1, j, 0]) - updatecoeffsE[materialEz, 2] * (Hx[i, j, 0] - Hx[i, j - 1, 0])


#################################################
# Electric field updates - dispersive materials #
#################################################
cpdef void update_electric_dispersive_multipole_A(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    int maxpoles,
                    floattype_t[:, ::1] updatecoeffsE,
                    complextype_t[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    complextype_t[:, :, :, ::1] Tx,
                    complextype_t[:, :, :, ::1] Ty,
                    complextype_t[:, :, :, ::1] Tz,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez,
                    floattype_t[:, :, ::1] Hx,
                    floattype_t[:, :, ::1] Hy,
                    floattype_t[:, :, ::1] Hz
            ):
    """This function updates the electric field components when dispersive materials (with multiple poles) are present.

    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """

    cdef Py_ssize_t i, j, k, pole
    cdef int material
    cdef float phi = 0

    # Ex component
    if ny != 1 or nz != 1:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        phi = phi + updatecoeffsdispersive[material, pole * 3].real * Tx[pole, i, j, k].real
                        Tx[pole, i, j, k] = updatecoeffsdispersive[material, 1 + (pole * 3)] * Tx[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] * Ex[i, j, k]
                    Ex[i, j, k] = updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        phi = phi + updatecoeffsdispersive[material, pole * 3].real * Ty[pole, i, j, k].real
                        Ty[pole, i, j, k] = updatecoeffsdispersive[material, 1 + (pole * 3)] * Ty[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] * Ey[i, j, k]
                    Ey[i, j, k] = updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    phi = 0
                    for pole in range(maxpoles):
                        phi = phi + updatecoeffsdispersive[material, pole * 3].real * Tz[pole, i, j, k].real
                        Tz[pole, i, j, k] = updatecoeffsdispersive[material, 1 + (pole * 3)] * Tz[pole, i, j, k] + updatecoeffsdispersive[material, 2 + (pole * 3)] * Ez[i, j, k]
                    Ez[i, j, k] = updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi



cpdef void update_electric_dispersive_multipole_B(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    int maxpoles,
                    complextype_t[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    complextype_t[:, :, :, ::1] Tx,
                    complextype_t[:, :, :, ::1] Ty,
                    complextype_t[:, :, :, ::1] Tz,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez
            ):
    """This function updates a temporary dispersive material array when disperisive materials (with multiple poles) are present.

    Args:
        nx, ny, nz (int): Grid size in cells
        maxpoles (int): Maximum number of poles
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """

    cdef Py_ssize_t i, j, k, pole
    cdef int material

    # Ex component
    if ny != 1 or nz != 1:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    for pole in range(maxpoles):
                        Tx[pole, i, j, k] = Tx[pole, i, j, k] - updatecoeffsdispersive[material, 2 + (pole * 3)] * Ex[i, j, k]

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    for pole in range(maxpoles):
                        Ty[pole, i, j, k] = Ty[pole, i, j, k] - updatecoeffsdispersive[material, 2 + (pole * 3)] * Ey[i, j, k]

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    for pole in range(maxpoles):
                        Tz[pole, i, j, k] = Tz[pole, i, j, k] - updatecoeffsdispersive[material, 2 + (pole * 3)] * Ez[i, j, k]


cpdef void update_electric_dispersive_1pole_A(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    floattype_t[:, ::1] updatecoeffsE,
                    complextype_t[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    complextype_t[:, :, :, ::1] Tx,
                    complextype_t[:, :, :, ::1] Ty,
                    complextype_t[:, :, :, ::1] Tz,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez,
                    floattype_t[:, :, ::1] Hx,
                    floattype_t[:, :, ::1] Hy,
                    floattype_t[:, :, ::1] Hz
            ):
    """This function updates the electric field components when dispersive materials (with 1 pole) are present.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E, H (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """

    cdef Py_ssize_t i, j, k
    cdef int material
    cdef float phi = 0

    # Ex component
    if ny != 1 or nz != 1:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    phi = updatecoeffsdispersive[material, 0].real * Tx[0, i, j, k].real
                    Tx[0, i, j, k] = updatecoeffsdispersive[material, 1] * Tx[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ex[i, j, k]
                    Ex[i, j, k] = updatecoeffsE[material, 0] * Ex[i, j, k] + updatecoeffsE[material, 2] * (Hz[i, j, k] - Hz[i, j - 1, k]) - updatecoeffsE[material, 3] * (Hy[i, j, k] - Hy[i, j, k - 1]) - updatecoeffsE[material, 4] * phi

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    phi = updatecoeffsdispersive[material, 0].real * Ty[0, i, j, k].real
                    Ty[0, i, j, k] = updatecoeffsdispersive[material, 1] * Ty[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ey[i, j, k]
                    Ey[i, j, k] = updatecoeffsE[material, 0] * Ey[i, j, k] + updatecoeffsE[material, 3] * (Hx[i, j, k] - Hx[i, j, k - 1]) - updatecoeffsE[material, 1] * (Hz[i, j, k] - Hz[i - 1, j, k]) - updatecoeffsE[material, 4] * phi

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    phi = updatecoeffsdispersive[material, 0].real * Tz[0, i, j, k].real
                    Tz[0, i, j, k] = updatecoeffsdispersive[material, 1] * Tz[0, i, j, k] + updatecoeffsdispersive[material, 2] * Ez[i, j, k]
                    Ez[i, j, k] = updatecoeffsE[material, 0] * Ez[i, j, k] + updatecoeffsE[material, 1] * (Hy[i, j, k] - Hy[i - 1, j, k]) - updatecoeffsE[material, 2] * (Hx[i, j, k] - Hx[i, j - 1, k]) - updatecoeffsE[material, 4] * phi


cpdef void update_electric_dispersive_1pole_B(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    complextype_t[:, ::1] updatecoeffsdispersive,
                    np.uint32_t[:, :, :, ::1] ID,
                    complextype_t[:, :, :, ::1] Tx,
                    complextype_t[:, :, :, ::1] Ty,
                    complextype_t[:, :, :, ::1] Tz,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez
            ):
    """This function updates a temporary dispersive material array when disperisive materials (with 1 pole) are present.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, T, ID, E (memoryviews): Access to update coeffients, temporary, ID and field component arrays
    """

    cdef Py_ssize_t i, j, k
    cdef int material

    # Ex component
    if ny != 1 or nz != 1:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(1, nz):
                    material = ID[0, i, j, k]
                    Tx[0, i, j, k] = Tx[0, i, j, k] - updatecoeffsdispersive[material, 2] * Ex[i, j, k]

    # Ey component
    if nx != 1 or nz != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(1, nz):
                    material = ID[1, i, j, k]
                    Ty[0, i, j, k] = Ty[0, i, j, k] - updatecoeffsdispersive[material, 2] * Ey[i, j, k]

    # Ez component
    if nx != 1 or ny != 1:
        for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(1, ny):
                for k in range(0, nz):
                    material = ID[2, i, j, k]
                    Tz[0, i, j, k] = Tz[0, i, j, k] - updatecoeffsdispersive[material, 2] * Ez[i, j, k]


##########################
# Magnetic field updates #
##########################
cpdef void update_magnetic(
                    int nx,
                    int ny,
                    int nz,
                    int nthreads,
                    floattype_t[:, ::1] updatecoeffsH,
                    np.uint32_t[:, :, :, ::1] ID,
                    floattype_t[:, :, ::1] Ex,
                    floattype_t[:, :, ::1] Ey,
                    floattype_t[:, :, ::1] Ez,
                    floattype_t[:, :, ::1] Hx,
                    floattype_t[:, :, ::1] Hy,
                    floattype_t[:, :, ::1] Hz
            ):
    """This function updates the magnetic field components.

    Args:
        nx, ny, nz (int): Grid size in cells
        nthreads (int): Number of threads to use
        updatecoeffs, ID, E, H (memoryviews): Access to update coeffients, ID and field component arrays
    """

    cdef Py_ssize_t i, j, k
    cdef int materialHx, materialHy, materialHz

    # 2D
    if nx == 1 or ny == 1 or nz == 1:
        # Hx component
        if ny == 1 or nz == 1:
            for i in prange(1, nx, nogil=True, schedule='static', num_threads=nthreads):
                for j in range(0, ny):
                    for k in range(0, nz):
                        materialHx = ID[3, i, j, k]
                        Hx[i, j, k] = updatecoeffsH[materialHx, 0] * Hx[i, j, k] - updatecoeffsH[materialHx, 2] * (Ez[i, j + 1, k] - Ez[i, j, k]) + updatecoeffsH[materialHx, 3] * (Ey[i, j, k + 1] - Ey[i, j, k])

        # Hy component
        if nx == 1 or nz == 1:
            for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
                for j in range(1, ny):
                    for k in range(0, nz):
                        materialHy = ID[4, i, j, k]
                        Hy[i, j, k] = updatecoeffsH[materialHy, 0] * Hy[i, j, k] - updatecoeffsH[materialHy, 3] * (Ex[i, j, k + 1] - Ex[i, j, k]) + updatecoeffsH[materialHy, 1] * (Ez[i + 1, j, k] - Ez[i, j, k])

        # Hz component
        if nx == 1 or ny == 1:
            for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
                for j in range(0, ny):
                    for k in range(1, nz):
                        materialHz = ID[5, i, j, k]
                        Hz[i, j, k] = updatecoeffsH[materialHz, 0] * Hz[i, j, k] - updatecoeffsH[materialHz, 1] * (Ey[i + 1, j, k] - Ey[i, j, k]) + updatecoeffsH[materialHz, 2] * (Ex[i, j + 1, k] - Ex[i, j, k])
    # 3D
    else:
        for i in prange(0, nx, nogil=True, schedule='static', num_threads=nthreads):
            for j in range(0, ny):
                for k in range(0, nz):
                    materialHx = ID[3, i + 1, j, k]
                    materialHy = ID[4, i, j + 1, k]
                    materialHz = ID[5, i, j, k + 1]
                    Hx[i + 1, j, k] = updatecoeffsH[materialHx, 0] * Hx[i + 1, j, k] - updatecoeffsH[materialHx, 2] * (Ez[i + 1, j + 1, k] - Ez[i + 1, j, k]) + updatecoeffsH[materialHx, 3] * (Ey[i + 1, j, k + 1] - Ey[i + 1, j, k])
                    Hy[i, j + 1, k] = updatecoeffsH[materialHy, 0] * Hy[i, j + 1, k] - updatecoeffsH[materialHy, 3] * (Ex[i, j + 1, k + 1] - Ex[i, j + 1, k]) + updatecoeffsH[materialHy, 1] * (Ez[i + 1, j + 1, k] - Ez[i, j + 1, k])
                    Hz[i, j, k + 1] = updatecoeffsH[materialHz, 0] * Hz[i, j, k + 1] - updatecoeffsH[materialHz, 1] * (Ey[i + 1, j, k + 1] - Ey[i, j, k + 1]) + updatecoeffsH[materialHz, 2] * (Ex[i, j + 1, k + 1] - Ex[i, j, k + 1])


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
    #It may seem like we are updated inside the pmls, and we are ! But those fields will be overwritten by the update
    # of the pml fields after this function

    cdef double complex*** Er = alloc_and_copy_complex3D(Er_np)
    cdef double complex*** Ephi = alloc_and_copy_complex3D(Ephi_np)
    cdef double complex*** Ez = alloc_and_copy_complex3D(Ez_np)
    cdef double complex*** Hr = alloc_and_copy_complex3D(Hr_np)
    cdef double complex*** Hphi = alloc_and_copy_complex3D(Hphi_np)
    cdef double complex*** Hz = alloc_and_copy_complex3D(Hz_np)

    taille_i_fields = Er_np.shape[0]
    taille_j_fields = Er_np.shape[1]
    print(updatecoeffsE[ID[2,1,0,50],0] * Ez[1][0][50])
    print((Hphi[1][0][50]*(1+0.5) - Hphi[1-1][0][50]*(1-0.5))/(1-0.5) * updatecoeffsE[ID[2,1,0,50], 1])
    print((Hphi[1][0][50] - Hphi[1][0][50-1]) * updatecoeffsE[ID[0,1,0,50], 3])
    for i in prange(1, nr, nogil= True, schedule= 'static', num_threads= nthreads):
        for k in range(1, nz):
            materialEr = ID[0,i,0,k]
            materialEphi = ID[1,i,0,k]
            materialEz = ID[2,i,0,k]

            Er[i][0][k] = I * m / (i+0.5) * Hz[i][0][k] * updatecoeffsE[materialEr, 1] - (Hphi[i][0][k] - Hphi[i][0][k-1]) * updatecoeffsE[materialEr, 3] + Er[i][0][k] * updatecoeffsE[materialEr, 0]

            Ephi[i][0][k] = (Hr[i][0][k] - Hr[i][0][k-1]) * updatecoeffsE[materialEphi,3] - (Hz[i][0][k] - Hz[i-1][0][k]) * updatecoeffsE[materialEphi, 1] + Ephi[i][0][k] * updatecoeffsE[materialEphi, 0]

            Ez[i][0][k] = (Hphi[i][0][k]*(i+0.5) - Hphi[i-1][0][k]*(i-0.5))/(i-0.5) * updatecoeffsE[materialEz, 1] - I * m / (i+0.5) * Hr[i][0][k] * updatecoeffsE[materialEz, 1] + Ez[i][0][k] * updatecoeffsE[materialEz, 0]

    cdef floattype_t pi = np.pi
    cdef floattype_t e = np.exp(1)
    if  np.abs(m) == 1:
        for k in prange(1, nz, nogil=True, schedule= 'static', num_threads= nthreads):
            Er[0][0][k] = Er[2][0][k] * e**(I * m * pi)
            Ephi[0][0][k] = Ephi[2][0][k] * e**(I * m * pi)
            Ez[0][0][k] = 0
    elif m == 0:
        for k in prange(1, nz, nogil=True, schedule= 'static', num_threads= nthreads):
            Er[0][0][k] = Er[2][0][k]
            Ephi[0][0][k] = 0
            Ez[0][0][k] = Ez[2][0][k]
         
    else:
        for k in prange(1, nz, nogil=True, schedule= 'static', num_threads= nthreads):
            Er[0][0][k] = 0
            Ephi[0][0][k] = 0
            Ez[0][0][k] = 0

    # for i in prange(1, nr, nogil= True, schedule= 'static', num_threads= nthreads):
    #     for k in range(1, nz):
    #         materialEr = ID[0,i,0,k]
    #         materialEphi = ID[1,i,0,k]
    #         materialEz = ID[2,i,0,k]

    #         Er[i][0][k] = Er[i][0][k] * updatecoeffsE[materialEr, 0] + I * m / (i+0.5) * Hz[i][0][k] * updatecoeffsE[materialEr, 1] - (Hphi[i][0][k] - Hphi[i][0][k-1]) * updatecoeffsE[materialEr, 3]

    #         Ephi[i][0][k] = Ephi[i][0][k] * updatecoeffsE[materialEphi, 0] + (Hr[i][0][k] - Hr[i][0][k-1]) * updatecoeffsE[materialEphi,3] - (Hz[i][0][k] - Hz[i-1][0][k]) * updatecoeffsE[materialEphi, 1]

    #         Ez[i][0][k] = Ez[i][0][k] * updatecoeffsE[materialEz, 0] + (Hphi[i][0][k]*(i+0.5) - Hphi[i-1][0][k]*(i-0.5))/i * updatecoeffsE[materialEz, 1] - I * m / i * Hr[i][0][k] * updatecoeffsE[materialEz, 1]

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


    for i in prange(0, nr, nogil= True, schedule= 'static', num_threads= nthreads):
        for k in range(0, nz):
            materialHr = ID[3,i,0,k]
            materialHphi = ID[4,i,0,k]
            materialHz = ID[5,i,0,k]

            Hr[i][0][k] = - I * m / (i+1) * Ez[i+1][0][k] * updatecoeffsH[materialHr, 1] + (Ephi[i][0][k+1] - Ephi[i][0][k]) * updatecoeffsH[materialHr, 3] + Hr[i][0][k] * updatecoeffsH[materialHr, 0]

            Hphi[i][0][k] = (Ez[i+1][0][k] - Ez[i][0][k]) * updatecoeffsH[materialHphi,1] - (Er[i][0][k+1] - Er[i][0][k]) * updatecoeffsH[materialHphi, 3] + Hphi[i][0][k] * updatecoeffsH[materialHphi, 0]

            Hz[i][0][k] = - (Ephi[i+1][0][k]*(i+1) - Ephi[i][0][k]*i)/(i+0.5) * updatecoeffsH[materialHz, 1] + I * m / (i+0.5) * Er[i][0][k] * updatecoeffsH[materialHz, 1] + Hz[i][0][k] * updatecoeffsH[materialHz, 0]

    if m == 0:
        for k in prange(1, nz, nogil=True, schedule= 'static', num_threads= nthreads):
            Hr[0][0][k] = 0
    elif np.abs(m) != 1:
        for k in prange(0,nz,nogil=True, schedule= 'static', num_threads=nthreads):
            Hr[0][0][k] = 0
            Hphi[0][0][k] = 0
            Hz[0][0][k] = 0

    # for i in prange(1, nr, nogil= True, schedule= 'static', num_threads= nthreads):
    #     for k in range(1, nz):
    #         materialHr = ID[3,i,0,k]
    #         materialHphi = ID[4,i,0,k]
    #         materialHz = ID[5,i,0,k]

    #         Hr[i][0][k] = Hr[i][0][k] * updatecoeffsH[materialHr, 0] + (Ephi[i][0][k+1] - Ephi[i][0][k]) * updatecoeffsH[materialHr, 3] - I * m / i * Ez[i+1][0][k] * updatecoeffsH[materialHr, 1] 

    #         Hphi[i][0][k] = Hphi[i][0][k] * updatecoeffsH[materialHphi, 0] + (Ez[i+1][0][k] - Ez[i][0][k]) * updatecoeffsH[materialHphi,1] - (Er[i][0][k+1] - Er[i][0][k]) * updatecoeffsH[materialHphi, 3]

    #         Hz[i][0][k] = Hz[i][0][k] * updatecoeffsH[materialHz, 0] + I * m / (i+0.5) * Er[i][0][k] * updatecoeffsH[materialHz, 1] - (Ephi[i+1][0][k]*(i+1) - Ephi[i][0][k]*i)/(i+0.5) * updatecoeffsH[materialHz, 1]

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