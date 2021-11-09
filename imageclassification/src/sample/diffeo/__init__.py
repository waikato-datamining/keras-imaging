"""
Module for working with diffeomorphisms of images.

As from the paper 'Relative stability toward diffeomorphisms indicates performance in deep nets
(Petrini, L. et al., 2021) [https://arxiv.org/abs/2105.02468]', and its related code-bases,
https://github.com/pcsl-epfl/diffeomorphism and https://github.com/leonardopetrini/diffeo-sota.
"""
from ._deform_image_with_diffeomorphism import deform_image_with_diffeomorphism
from ._displacement_field import displacement_field
from ._Equation_2 import Equation_2
from ._Equation_2_components import Equation_2_components
from ._Interpolation import Interpolation
from ._noise import noise
from ._offset_norm import offset_norm
from ._relative_stability_to_diffeomorphisms import relative_stability_to_diffeomorphisms
from ._temperature_range import temperature_range