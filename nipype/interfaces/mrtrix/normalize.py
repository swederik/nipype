# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Change directory to provide relative paths for doctests
    >>> import os
    >>> filepath = os.path.dirname( os.path.realpath( __file__ ) )
    >>> datadir = os.path.realpath(os.path.join(filepath, '../../testing/data'))
    >>> os.chdir(datadir)

"""

from nipype.interfaces.base import CommandLineInputSpec, CommandLine, traits, TraitedSpec, File
from nipype.utils.filemanip import split_filename
import os, os.path as op
  
class NormalizeTracksInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-3,
        desc='the input MRtrix (.tck) track file')
    transform_image = File(exists=True, argstr='%s', mandatory=True, position=-2,
        desc='the image containing the transform.')        
    out_filename = File(genfile=True, argstr='%s', position=-1, desc='Output normalized track file name')
    quiet = traits.Bool(argstr='-quiet', position=1, desc="Do not display information messages or progress status.")
    debug = traits.Bool(argstr='-debug', position=1, desc="Display debugging messages.")

class NormalizeTracksOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output normalized track file')

class NormalizeTracks(CommandLine):
    """
    Applies a normalisation map to a track file

    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> norm = mrt.NormalizeTracks()
    >>> norm.inputs.in_file = 'tracks.tck'
    >>> norm.inputs.transform_image = 'warp_image.nii'
    >>> norm.run()                                 # doctest: +SKIP
    """

    _cmd = 'normalise_tracks'
    input_spec=NormalizeTracksInputSpec
    output_spec=NormalizeTracksOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self._gen_outfilename())
        return outputs

    def _gen_filename(self, name):
        if name is 'out_filename':
            return self._gen_outfilename()
        else:
            return None
    def _gen_outfilename(self):
        _, name , _ = split_filename(self.inputs.in_file)
        return name + '_normalized'

class GenerateUnitWarpFieldInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='%s', mandatory=True, position=-2,
        desc='the input template image (.nii or .mif)')
    out_filename = File(genfile=True, argstr='%s', position=-1, desc='Output normalized track file name')
    quiet = traits.Bool(argstr='-quiet', position=1, desc="Do not display information messages or progress status.")
    debug = traits.Bool(argstr='-debug', position=1, desc="Display debugging messages.")

class GenerateUnitWarpFieldOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='the output normalized track file')


class GenerateUnitWarpField(CommandLine):
    """ Generates a warp field corresponding to a no-warp operation.
    
	This is useful to obtain the warp fields from other normalisation
    applications, by applying the warp of interest to the the warp field
	generated by this program.
          
    Example
    -------

    >>> import nipype.interfaces.mrtrix as mrt
    >>> warp = mrt.GenerateUnitWarpField()
    >>> warp.inputs.in_file = 'template.nii'
    >>> warp.run()                                 # doctest: +SKIP
    """

    _cmd = 'gen_unit_warp'
    input_spec=GenerateUnitWarpFieldInputSpec
    output_spec=GenerateUnitWarpFieldOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = op.abspath(self._gen_outfilename())
        return outputs

    def _gen_filename(self, name):
        if name is 'out_filename':
            return self._gen_outfilename()
        else:
            return None
    def _gen_outfilename(self):
        _, name , _ = split_filename(self.inputs.in_file)
        return name + '_warpfield.nii'
