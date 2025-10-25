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

from gprMax.constants import e0
from gprMax.constants import m0
from gprMax.constants import complextype


class Material(object):
    """Materials, their properties and update coefficients."""

    def __init__(self, numID, ID):
        """
        Args:
            numID (int): Numeric identifier of the material.
            ID (str): Name of the material.
        """

        self.numID = numID
        self.ID = ID
        self.type = ''
        # Default material averaging
        self.averagable = True

        # Default material constitutive parameters (free_space)
        self.er = 1.0
        self.se = 0.0
        self.mr = 1.0
        self.sm = 0.0

    def calculate_update_coeffsH(self, G):
        """Calculates the magnetic update coefficients of the material.

        Args:
            G (class): Grid class instance - holds essential parameters describing the model.
        """

        HA = (m0 * self.mr / G.dt) + 0.5 * self.sm
        HB = (m0 * self.mr / G.dt) - 0.5 * self.sm
        self.DA = HB / HA
        self.DBx = (1 / G.dx) * 1 / HA
        self.DBy = (1 / G.dy) * 1 / HA
        self.DBz = (1 / G.dz) * 1 / HA
        self.srcm = 1 / HA

    def calculate_update_coeffsE(self, G):
        """Calculates the electric update coefficients of the material.

        Args:
            G (class): Grid class instance - holds essential parameters
                    describing the model.
        """

        EA = (e0 * self.er / G.dt) + 0.5 * self.se
        EB = (e0 * self.er / G.dt) - 0.5 * self.se

        if self.ID == 'pec' or self.se == float('inf'):
            self.CA = 0
            self.CBx = 0
            self.CBy = 0
            self.CBz = 0
            self.srce = 0
        else:
            self.CA = EB / EA
            self.CBx = (1 / G.dx) * 1 / EA
            self.CBy = (1 / G.dy) * 1 / EA
            self.CBz = (1 / G.dz) * 1 / EA
            self.srce = 1 / EA

def process_materials(G):
    """
    Process complete list of materials - calculate update coefficients,
        store in arrays, and build text list of materials/properties

    Args:
        G (class): Grid class instance - holds essential parameters describing the model.

    Returns:
        materialsdata (list): List of material IDs, names, and properties to print a table.
    """


    materialsdata = [['\nID', '\nName', '\nType', '\neps_r', 'sigma\n[S/m]', '\nmu_r', 'sigma*\n[Ohm/m]', 'Dielectric\nsmoothable']]

    for material in G.materials:
        # Calculate update coefficients for material
        material.calculate_update_coeffsE(G)
        material.calculate_update_coeffsH(G)

        # Store all update coefficients together
        G.updatecoeffsE[material.numID, :] = material.CA, material.CBx, material.CBy, material.CBz, material.srce
        G.updatecoeffsH[material.numID, :] = material.DA, material.DBx, material.DBy, material.DBz, material.srcm

        # Construct information on material properties for printing table
        materialtext = []
        materialtext.append(str(material.numID))
        materialtext.append(material.ID[:50] if len(material.ID) > 50 else material.ID)
        materialtext.append(material.type)
        materialtext.append('{:g}'.format(material.er))
        materialtext.append('{:g}'.format(material.se))
        materialtext.append('{:g}'.format(material.mr))
        materialtext.append('{:g}'.format(material.sm))
        materialtext.append(material.averagable)
        materialsdata.append(materialtext)

    return materialsdata