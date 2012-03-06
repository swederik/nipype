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

def get_out_paths(in_paths, suffix='_pruned'):
	if isinstance(in_paths, list):
		out_paths = []
		for path in in_paths:
			_, name, ext = split_filename(path)
			out_paths.append(op.abspath(name + suffix + '.pck'))
		return out_paths
	else:
		_, name, ext = split_filename(in_paths)
		return op.abspath(name + suffix + '.pck')
    
class RemoveUnconnectedNodesInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, desc='Networks for the first group of subjects')
    by_group = traits.Bool(True, usedefault=True, desc='Option to require that nodes saved are connected in every graph input.')
    remove_all_unconnected = traits.Bool(True, usedefault=True, desc='Option to remove nodes based on the union of unconnected nodes from the entire group.')
    output_as_cff = traits.Bool(True, usedefault=True, desc='Option to save the output networks in a Connectome File Format (CFF) file')
    output_cff_file = File('connected.cff', usedefault=True, desc='The output networks saved in a single CFF file')

class RemoveUnconnectedNodesOutputSpec(TraitedSpec):
    connectome_file = File(desc='The output networks saved in a single CFF file')
    out_files = OutputMultiPath(File(desc='Output networks with only the connected nodes'))

class RemoveUnconnectedNodes(BaseInterface):
	"""
	Calculates and outputs the average network given a set of input NetworkX gpickle files

	Example
	-------

	>>> import nipype.interfaces.cmtk as cmtk
	>>> union = cmtk.RemoveUnconnectedNodes()
	>>> union.inputs.in_files = ['subj1.pck', 'subj2.pck'] # doctest: +SKIP
	>>> union.run()                 # doctest: +SKIP
	"""
	input_spec = RemoveUnconnectedNodesInputSpec
	output_spec = RemoveUnconnectedNodesOutputSpec

	def _run_interface(self, runtime):
		unconnected = []
		union_of_unconnected = []
		for idx, in_file in enumerate(self.inputs.in_files):
			out_path = get_out_paths(in_file)
			graph = nx.read_gpickle(in_file)			
			unconnected = nx.isolates(graph)
			iflogger.info('For file: {i}, {n} nodes are unconnected'.format(i=in_file.split('- ')[-1], n=len(unconnected)))
			if idx == 0:
				union_of_unconnected = unconnected
				intersection_of_unconnected = unconnected
			new_graph = graph.copy()
			if not self.inputs.by_group:
				new_graph.remove_nodes_from(unconnected)
				mapping=dict(zip(new_graph.nodes(),range(1,new_graph.number_of_nodes()+1)))
				new_graph=nx.relabel_nodes(new_graph,mapping)
				nx.write_gpickle(new_graph, out_path)
				iflogger.info('Saving pruned output network as {out}'.format(out=out_path))
				iflogger.info('{N} nodes removed: {x}'.format(N=len(unconnected), x=unconnected))
			elif self.inputs.remove_all_unconnected:
				union_of_unconnected = list(set(union_of_unconnected).union(set(unconnected)))
			else:
				intersection_of_unconnected = list(set(union_of_unconnected).intersection( set(unconnected) ))

		if self.inputs.by_group:
			for in_file in self.inputs.in_files:
				out_path = get_out_paths(in_file)
				graph = nx.read_gpickle(in_file)
				new_graph = graph.copy()
				if self.inputs.remove_all_unconnected:
					remove = union_of_unconnected
				else:
					remove = intersection_of_unconnected
				iflogger.info('Saving pruned output network as {out}'.format(out=out_path))
				new_graph.remove_nodes_from(remove)
				mapping=dict(zip(new_graph.nodes(),range(1,new_graph.number_of_nodes()+1)))
				new_graph=nx.relabel_nodes(new_graph,mapping)
				nx.write_gpickle(new_graph, out_path)

		if self.inputs.output_as_cff:
			convert = CFFConverter()
			out_paths = get_out_paths(self.inputs.in_files)
			convert.inputs.gpickled_networks = out_paths
			iflogger.info(out_paths)
			convert.inputs.out_file = op.abspath(self.inputs.output_cff_file)
			convert.run()
			iflogger.info('Saving output CFF file as {out}'.format(out=op.abspath(self.inputs.output_cff_file)))
		
		iflogger.info('{N} nodes removed - pruned by the union of unconnected nodes: {x}'.format(N=len(remove), x=remove))
		return runtime

	def _list_outputs(self):
		outputs = self.output_spec().get()
		out_paths = get_out_paths(self.inputs.in_files)
		outputs['out_files'] = out_paths
		if self.inputs.output_as_cff:
			outputs['connectome_file'] = op.abspath(self.inputs.output_cff_file)
		return outputs

	def _gen_outfilename(self, name, ext):
		return name + '.' + ext


def remove_nodes_named(in_file, phrase='occipital', out_file='removed.pck'):
    in_ntwk = nx.read_gpickle(in_file)
    out_ntwk = in_ntwk.copy()
    nodes = in_ntwk.nodes_iter()
    count = 0
    for node in nodes:
        node_name = str(in_ntwk.node[node]['dn_name'])
        if node_name.rfind(phrase) >= 0:
            iflogger.info(node_name)
            count += 1
            out_ntwk.remove_node(node)
    iflogger.info('{N} nodes removed with "{n}" in their name'.format(N=count, n=phrase))
    mapping=dict(zip(out_ntwk.nodes(),range(1,out_ntwk.number_of_nodes()+1)))
    out_ntwk=nx.relabel_nodes(out_ntwk,mapping)
    iflogger.info('Writing network as {o}'.format(o=out_file))
    nx.write_gpickle(out_ntwk, op.abspath(out_file))
    return out_file
    
class RemoveNodesByPhraseInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, desc='Networks for node removal subjects')
    phrases = traits.List(['occipital'], usedefault=True, desc='Network node names to remove from the network')
    output_as_cff = traits.Bool(True, usedefault=True, desc='Option to save the output networks in a Connectome File Format (CFF) file')
    output_cff_file = File('removed.cff', usedefault=True, desc='The output networks saved in a single CFF file')

class RemoveNodesByPhraseOutputSpec(TraitedSpec):
    connectome_file = File(desc='The output networks saved in a single CFF file')
    out_files = OutputMultiPath(File(desc='Output networks with only the connected nodes'))

class RemoveNodesByPhrase(BaseInterface):
    """
    Removes nodes given a supplied list of node names or a that contain a word such as 'occipital'

    Example
    -------

    >>> import nipype.interfaces.cmtk as cmtk
    >>> rm = cmtk.RemoveNodesByPhrase()
    >>> rm.inputs.in_files = ['subj1.pck', 'subj2.pck'] # doctest: +SKIP
    >>> rm.run()                 # doctest: +SKIP
    """
    input_spec = RemoveNodesByPhraseInputSpec
    output_spec = RemoveNodesByPhraseOutputSpec

    def _run_interface(self, runtime):
        global out_paths
        out_paths = []
        for idx, phrase in enumerate(self.inputs.phrases):
            # Phrases are removed sequentially
            for in_file in self.inputs.in_files:
                if idx == 0:
                    out_path = get_out_paths(in_file)
                    remove_nodes_named(in_file, phrase, out_path)
                else:
                    remove_nodes_named(out_path, phrase, out_path)
                out_paths.append(out_path)
                        
        if self.inputs.output_as_cff:
            convert = CFFConverter()
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
            outputs['connectome_file'] = op.abspath(self.inputs.output_cff_file)
        return outputs

    def _gen_outfilename(self, name, ext):
        return name + '.' + ext
