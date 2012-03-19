from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec, traits,
                                    File, TraitedSpec, InputMultiPath,
                                    OutputMultiPath, isdefined)
from nipype.utils.filemanip import split_filename
import os, os.path as op
import networkx as nx
import pickle
from nipype.interfaces.cmtk.convert import CFFConverter
import logging

logging.basicConfig()
iflogger = logging.getLogger('interface')

import os, os.path as op

def create_rotation_frames(out_folder, out_name, frames=4, ext='.png'):
	for frame in range(1,frames+1):
		f = mlab.gcf()
		f.scene.camera.azimuth(1)
		name = '%04d' %frame
		out_frame = op.join(out_folder, out_name) + name + ext
		f.scene.render()
		mlab.savefig(out_frame)

class RotationMovieInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Networks for node removal subjects')
    number_of_frames = traits.Int(360, usedefault=True, desc='Number of frames to generate')
    x_degrees = traits.Int(360, usedefault=True, desc='Degrees to rotate around the x axis')
    y_degrees = traits.Int(0, usedefault=True, desc='Degrees to rotate around the y axis')
    z_degrees = traits.Int(0, usedefault=True, desc='Degrees to rotate around the z axis')
    output_as_mpeg = traits.Bool(False, usedefault=True, desc='Option to save the output networks in an mpeg movie file')
    output_mpeg_file = File('rotation.mpg', usedefault=True, desc='The output images saved as an mpeg movie file')

class RotationMovieOutputSpec(TraitedSpec):
    mpeg_file = File(desc='The output networks saved in a single mpeg file')
    out_files = OutputMultiPath(File(desc='Output sequence of images'))

class RotationMovie(BaseInterface):
    """
    Creates a movie by plotting single frames and combining them using ffmpeg

    Example
    -------

    >>> import nipype.interfaces.connectomeviewer as cv
    >>> rm = cv.RotationMovie()
    >>> rm.inputs.in_files = ['subj1.pck', 'subj2.pck'] # doctest: +SKIP
    >>> rm.run()                 # doctest: +SKIP
    """
    input_spec = RotationMovieInputSpec
    output_spec = RotationMovieOutputSpec

    def _run_interface(self, runtime):
        global out_paths
        out_paths = []
        create_rotation_frames(out_directory, 'const1', 360)
        for in_file in self.inputs.in_files:
            if idx == 0:
                out_path = get_out_paths(in_file)
                remove_nodes_named(in_file, phrase, out_path)
            else:
                remove_nodes_named(out_path, phrase, out_path)
            out_paths.append(out_path)
                        
        if self.inputs.output_as_cff:
            out_paths = get_out_paths(self.inputs.in_files)
            convert.inputs.gpickled_networks = out_paths
            iflogger.info(out_paths)
            convert.inputs.out_file = op.abspath(self.inputs.output_cff_file)
            convert.run()
            iflogger.info('Saving output CFF file as {out}'.format(out=op.abspath(self.inputs.output_cff_file)))
        
        isolate_list = []
        for idx, out_file in enumerate(out_paths):
            graph = nx.read_gpickle(out_file)
            n_isolates = len(nx.isolates(graph))
            iflogger.info('File: {f} has {n} unconnected nodes'.format(f=out_file, n=n_isolates))
            isolate_list.append(n_isolates)
        iflogger.info(isolate_list)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        out_paths = get_out_paths(self.inputs.in_files)
        outputs['out_files'] = out_paths
        if self.inputs.output_as_cff:
            outputs['mpeg_file'] = op.abspath(self.inputs.output_cff_file)
        return outputs        
