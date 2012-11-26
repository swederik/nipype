# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
    Change directory to provide relative paths for doctests
    >>> import os
    >>> filepath = os.path.dirname( os.path.realpath( __file__ ) )
    >>> datadir = os.path.realpath(os.path.join(filepath, '../../testing/data'))
    >>> os.chdir(datadir)

"""

from nipype.interfaces.base import (BaseInterface, BaseInterfaceInputSpec, traits,
                                    File, TraitedSpec, InputMultiPath,
                                    OutputMultiPath, isdefined)
from nipype.utils.filemanip import split_filename
import os, os.path as op
import numpy as np
import networkx as nx
import scipy.io as sio
import pickle
from nipype.utils.misc import package_check
from nipype.interfaces.cmtk.convert import CFFConverter
import warnings

from ... import logging
iflogger = logging.getLogger('interface')

have_cmp = True
try:
    package_check('cmp')
except Exception, e:
    have_cmp = False
    warnings.warn('cmp not installed')
else:
    import cmp


def read_unknown_ntwk(ntwk):
	if not isinstance(ntwk, nx.classes.graph.Graph):
		path, name, ext = split_filename(ntwk)
		if ext == '.pck':
			ntwk = nx.read_gpickle(ntwk)
		elif ext == '.graphml':
			ntwk = nx.read_graphml(ntwk)
	return ntwk


def remove_all_edges(ntwk):
    ntwktmp = ntwk.copy()
    edges = ntwktmp.edges_iter()
    for edge in edges:
        ntwk.remove_edge(edge[0], edge[1])
    return ntwk


def fix_keys_for_gexf(orig):
    """
    GEXF Networks can be read in Gephi, however, the keys for the node and edge IDs must be converted to strings
    """
    import networkx as nx
    ntwk = nx.Graph()
    nodes = orig.nodes_iter()
    edges = orig.edges_iter()
    for node in nodes:
        newnodedata = {}
        newnodedata.update(orig.node[node])
        if orig.node[node].has_key('dn_fsname'):
			newnodedata['label'] = orig.node[node]['dn_fsname']
        ntwk.add_node(str(node), newnodedata)
        if ntwk.node[str(node)].has_key('dn_position') and newnodedata.has_key('dn_position'):
            ntwk.node[str(node)]['dn_position'] = str(newnodedata['dn_position'])
    for edge in edges:
        data = {}
        data = orig.edge[edge[0]][edge[1]]
        ntwk.add_edge(str(edge[0]), str(edge[1]), data)
        if ntwk.edge[str(edge[0])][str(edge[1])].has_key('fiber_length_mean'):
            ntwk.edge[str(edge[0])][str(edge[1])]['fiber_length_mean'] = str(data['fiber_length_mean'])
        if ntwk.edge[str(edge[0])][str(edge[1])].has_key('fiber_length_std'):
            ntwk.edge[str(edge[0])][str(edge[1])]['fiber_length_std'] = str(data['fiber_length_std'])
        if ntwk.edge[str(edge[0])][str(edge[1])].has_key('number_of_fibers'):
            ntwk.edge[str(edge[0])][str(edge[1])]['number_of_fibers'] = str(data['number_of_fibers'])
        if ntwk.edge[str(edge[0])][str(edge[1])].has_key('value'):
            ntwk.edge[str(edge[0])][str(edge[1])]['value'] = str(data['value'])
    return ntwk


def add_dicts_by_key(in_dict1, in_dict2, subtract=False):
    """
    Combines two dictionaries and adds the values for those keys that are shared
    """
    both = {}
    for key1 in in_dict1:
        for key2 in in_dict2:
            if key1 == key2:
                if subtract:
                    both[key1] = in_dict2[key2] - in_dict1[key1]
                else:
                    both[key1] = in_dict1[key1] + in_dict2[key2]
    return both


def average_networks(in_files, ntwk_res_file, group_id):
    """
    Sums the edges of input networks and divides by the number of networks
    Writes the average network as .pck and .gexf and returns the name of the written networks
    """
    import networkx as nx
    import os.path as op
    iflogger.info("Creating average network for group: {grp}".format(grp=group_id))
    matlab_network_list = []
    if len(in_files) == 1:
        avg_ntwk = read_unknown_ntwk(in_files[0])
    else:
        count_to_keep_edge = np.round(float(len(in_files)) / 2)
        iflogger.info("Number of networks: {L}, an edge must occur in at least {c} to remain in the average network".format(L=len(in_files), c=count_to_keep_edge))
        ntwk_res_file = read_unknown_ntwk(ntwk_res_file)
        iflogger.info("{n} Nodes found in network resolution file".format(n=ntwk_res_file.number_of_nodes()))
        ntwk = remove_all_edges(ntwk_res_file)
        counting_ntwk = ntwk.copy()
        # Sums all the relevant variables
        for index, subject in enumerate(in_files):
            tmp = nx.read_gpickle(subject)
            iflogger.info('File {s} has {n} edges'.format(s=subject, n=tmp.number_of_edges()))
            edges = tmp.edges_iter()
            for edge in edges:
                data = {}
                data = tmp.edge[edge[0]][edge[1]]
                data['count'] = 1
                if ntwk.has_edge(edge[0], edge[1]):
                    current = {}
                    current = ntwk.edge[edge[0]][edge[1]]
                    data = add_dicts_by_key(current, data)
                ntwk.add_edge(edge[0], edge[1], data)
            nodes = tmp.nodes_iter()
            for node in nodes:
                data = {}
                data = ntwk.node[node]
                if tmp.node[node].has_key('value'):
                    data['value'] = data['value'] + tmp.node[node]['value']
                ntwk.add_node(node, data)

        # Divides each value by the number of files
        nodes = ntwk.nodes_iter()
        edges = ntwk.edges_iter()
        iflogger.info('Total network has {n} edges'.format(n=ntwk.number_of_edges()))
        avg_ntwk = nx.Graph()
        newdata = {}
        for node in nodes:
            data = ntwk.node[node]
            newdata = data
            if data.has_key('value'):
                newdata['value'] = data['value'] / len(in_files)
                ntwk.node[node]['value'] = newdata
            avg_ntwk.add_node(node, newdata)

        edge_dict = {}
        edge_dict['count'] = np.zeros((avg_ntwk.number_of_nodes(), avg_ntwk.number_of_nodes()))
        for edge in edges:
            data = ntwk.edge[edge[0]][edge[1]]
            if ntwk.edge[edge[0]][edge[1]]['count'] >= count_to_keep_edge:
                for key in data.keys():
                    if not key == 'count':
                        data[key] = data[key] / len(in_files)
                ntwk.edge[edge[0]][edge[1]] = data
                avg_ntwk.add_edge(edge[0],edge[1],data)
            edge_dict['count'][edge[0]-1][edge[1]-1] = ntwk.edge[edge[0]][edge[1]]['count']

        iflogger.info('After thresholding, the average network has has {n} edges'.format(n=avg_ntwk.number_of_edges()))

        avg_edges = avg_ntwk.edges_iter()
        for edge in avg_edges:
            data = avg_ntwk.edge[edge[0]][edge[1]]
            for key in data.keys():
                if not key == 'count':
                    edge_dict[key] = np.zeros((avg_ntwk.number_of_nodes(), avg_ntwk.number_of_nodes()))
                    edge_dict[key][edge[0]-1][edge[1]-1] = data[key]

        for key in edge_dict.keys():
            tmp = {}
            network_name = group_id + '_' + key + '_average.mat'
            matlab_network_list.append(op.abspath(network_name))
            tmp[key] = edge_dict[key]
            sio.savemat(op.abspath(network_name), tmp)
            iflogger.info('Saving average network for key: {k} as {out}'.format(k=key, out=op.abspath(network_name)))

    # Writes the networks and returns the name
    network_name = group_id + '_average.pck'
    nx.write_gpickle(avg_ntwk, op.abspath(network_name))
    iflogger.info('Saving average network as {out}'.format(out=op.abspath(network_name)))
    avg_ntwk = fix_keys_for_gexf(avg_ntwk)
    network_name = group_id + '_average.gexf'
    nx.write_gexf(avg_ntwk, op.abspath(network_name))
    iflogger.info('Saving average network as {out}'.format(out=op.abspath(network_name)))
    return network_name, matlab_network_list


def compute_node_measures(ntwk, calculate_cliques=False):
    """
    These return node-based measures
    """
    iflogger.info('Computing node measures:')
    measures = {}
    iflogger.info('...Computing degree...')
    measures['degree'] = np.array(ntwk.degree().values())
    iflogger.info('...Computing load centrality...')
    measures['load_centrality'] = np.array(nx.load_centrality(ntwk).values())
    iflogger.info('...Computing betweenness centrality...')
    measures['betweenness_centrality'] = np.array(nx.betweenness_centrality(ntwk).values())
    iflogger.info('...Computing degree centrality...')
    measures['degree_centrality'] = np.array(nx.degree_centrality(ntwk).values())
    iflogger.info('...Computing closeness centrality...')
    measures['closeness_centrality'] = np.array(nx.closeness_centrality(ntwk).values())
#    iflogger.info('...Computing eigenvector centrality...')
#    measures['eigenvector_centrality'] = np.array(nx.eigenvector_centrality(ntwk, max_iter=100000).values())
    iflogger.info('...Computing triangles...')
    measures['triangles'] = np.array(nx.triangles(ntwk).values())
    iflogger.info('...Computing clustering...')
    measures['clustering'] = np.array(nx.clustering(ntwk).values())
    iflogger.info('...Computing k-core number')
    measures['core_number'] = np.array(nx.core_number(ntwk).values())
    iflogger.info('...Identifying network isolates...')
    isolate_list = nx.isolates(ntwk)
    binarized = np.zeros((ntwk.number_of_nodes(), 1))
    for value in isolate_list:
        value = value - 1 # Zero indexing
        binarized[value] = 1
    measures['isolates'] = binarized
    if calculate_cliques:
        iflogger.info('...Calculating node clique number')
        measures['node_clique_number'] = np.array(nx.node_clique_number(ntwk).values())
        iflogger.info('...Computing number of cliques for each node...')
        measures['number_of_cliques'] = np.array(nx.number_of_cliques(ntwk).values())
    return measures


def compute_edge_measures(ntwk):
    """
    These return edge-based measures
    """
    iflogger.info('Computing edge measures:')
    measures = {}
    #iflogger.info('...Computing google matrix...' #Makes really large networks (500k+ edges))
    #measures['google_matrix'] = nx.google_matrix(ntwk)
    #iflogger.info('...Computing hub matrix...')
    #measures['hub_matrix'] = nx.hub_matrix(ntwk)
    #iflogger.info('...Computing authority matrix...')
    #measures['authority_matrix'] = nx.authority_matrix(ntwk)
    return measures


def compute_dict_measures(ntwk):
    """
    Returns a dictionary
    """
    iflogger.info('Computing measures which return a dictionary:')
    measures = {}
    iflogger.info('...Computing rich club coefficient...')
    measures['rich_club_coef'] = nx.rich_club_coefficient(ntwk)
    return measures


def compute_singlevalued_measures(ntwk, weighted=True, calculate_cliques=False):
    """
    Returns a single value per network
    """
    iflogger.info('Computing single valued measures:')
    measures = {}
    iflogger.info('...Computing degree assortativity (pearson number) ...')
    try:
        measures['degree_pearsonr'] = nx.degree_pearsonr(ntwk)
    except AttributeError: # For NetworkX 1.6
        measures['degree_pearsonr'] = nx.degree_pearson_correlation_coefficient(ntwk)
    iflogger.info('...Computing degree assortativity...')
    try:
        measures['degree_assortativity'] = nx.degree_assortativity(ntwk)
    except AttributeError:
        measures['degree_assortativity'] = nx.degree_assortativity_coefficient(ntwk)
    iflogger.info('...Computing transitivity...')
    measures['transitivity'] = nx.transitivity(ntwk)
    iflogger.info('...Computing number of connected_components...')
    measures['number_connected_components'] = nx.number_connected_components(ntwk)
    iflogger.info('...Computing graph density...')
    measures['graph_density'] = nx.density(ntwk)
    iflogger.info('...Recording number of edges...')
    measures['number_of_edges'] = nx.number_of_edges(ntwk)
    iflogger.info('...Recording number of nodes...')
    measures['number_of_nodes'] = nx.number_of_nodes(ntwk)
    iflogger.info('...Computing average clustering...')
    measures['average_clustering'] = nx.average_clustering(ntwk)
    if nx.is_connected(ntwk):
        iflogger.info('...Calculating average shortest path length...')
        measures['average_shortest_path_length'] = nx.average_shortest_path_length(ntwk, weighted)
    else:
        iflogger.info('...Calculating average shortest path length...')
        measures['average_shortest_path_length'] = nx.average_shortest_path_length(nx.connected_component_subgraphs(ntwk)[0], weighted)
    if calculate_cliques:
        iflogger.info('...Computing graph clique number...')
        measures['graph_clique_number'] = nx.graph_clique_number(ntwk) #out of memory error
    return measures


def compute_network_measures(ntwk):
    measures = {}
    #iflogger.info('Identifying k-core')
    #measures['k_core'] = nx.k_core(ntwk)
    #iflogger.info('Identifying k-shell')
    #measures['k_shell'] = nx.k_shell(ntwk)
    #iflogger.info('Identifying k-crust')
    #measures['k_crust'] = nx.k_crust(ntwk)
    return measures


def add_node_data(node_array, ntwk):
    node_ntwk = nx.Graph()
    newdata = {}
    for idx, data in ntwk.nodes_iter(data=True):
        if not int(idx) == 0:
            newdata['value'] = node_array[int(idx) - 1]
            data.update(newdata)
            node_ntwk.add_node(int(idx), data)
    return node_ntwk


def add_edge_data(edge_array, ntwk, above=0, below=0):
    edge_ntwk = ntwk.copy()
    data = {}
    for x, row in enumerate(edge_array):
        for y in range(0, np.max(np.shape(edge_array[x]))):
            if not edge_array[x, y] == 0:
				data['value'] = edge_array[x, y]
				if data['value'] <= below or data['value'] >= above:
					if edge_ntwk.has_edge(x + 1, y + 1):
						old_edge_dict = edge_ntwk.edge[x + 1][y + 1]
						edge_ntwk.remove_edge(x + 1, y + 1)
						data.update(old_edge_dict)
					edge_ntwk.add_edge(x + 1, y + 1, data)
    return edge_ntwk


def difference_graph(in_file1, in_file2, ntwk_res_file, keep_only_common_edges=-1):
    """
    Subtracts the edges from in_file1 from in_file2.
    
    Writes the difference network as 'in_file2-in_file1_difference' [.pck, .gexf]
    and returns the name of the written networks
    
    By default it will do basic subtraction as described (if keep_only_common_edges is left undefined). 
    The user can also specify  whether to retain only the common (keep_only_common_edges = True) or 
    uncommon (keep_only_common_edges = False) edges between the two graphs. 
    """
    import networkx as nx
    import os.path as op
    from nipype.utils.filemanip import split_filename
    iflogger.info("Creating difference network: {in2} - {in1}".format(in2=in_file2, in1=in_file1))
    
    _, name1, _ = split_filename(in_file1)
    _, name2, _ = split_filename(in_file2)
    ntwk1 = nx.read_gpickle(in_file1)
    iflogger.info('File {s} has {n} edges'.format(s=name1, n=ntwk1.number_of_edges()))
    ntwk2 = nx.read_gpickle(in_file2)
    iflogger.info('File {s} has {n} edges'.format(s=name2, n=ntwk2.number_of_edges()))

    ntwk_res_file = nx.read_gpickle(ntwk_res_file)
    iflogger.info("{n} Nodes found in network resolution file".format(n=ntwk_res_file.number_of_nodes()))
    
    diff_ntwk = nx.Graph()
    assert diff_ntwk.number_of_edges() == 0

    nodes = ntwk_res_file.nodes_iter()
    for node in nodes:
        data = {}
        dict1 = ntwk1.node[node]
        dict2 = ntwk2.node[node]
        if ntwk1.node[node].has_key('value') and ntwk2.node[node].has_key('value'):
            data['value'] = dict2['value'] - dict1['value']
        else:
            data = ntwk_res_file.node[node]
        diff_ntwk.add_node(node, data)

    edges1 = ntwk1.edges_iter()
    edges2 = ntwk2.edges_iter()
    for edge in edges2:
        data = {}
        dict2 = ntwk2.edge[edge[0]][edge[1]]
        if keep_only_common_edges == False:
            # Retain only edges in network 2 and not in network 1
            if not ntwk1.has_edge(edge[0], edge[1]):
                data = dict2
                diff_ntwk.add_edge(edge[0], edge[1], data)
                
        elif keep_only_common_edges == True:
            # Retain only edges in both networks and calculate the difference for each key at those edges
            if ntwk1.has_edge(edge[0], edge[1]):
                dict1 = ntwk1.edge[edge[0]][edge[1]]
                data = add_dicts_by_key(dict1, dict2, subtract=True)
                diff_ntwk.add_edge(edge[0], edge[1], data)

        elif keep_only_common_edges == -1:
            # Calculate the difference for each key at of all the edges in network 2
            if ntwk1.has_edge(edge[0], edge[1]):
                dict1 = ntwk1.edge[edge[0]][edge[1]]
                data = add_dicts_by_key(dict1, dict2, subtract=True)
                diff_ntwk.add_edge(edge[0], edge[1], data)
                
    if keep_only_common_edges == -1:
        # Calculate the difference for each key at all of the edges in network 1
        for edge in edges1:
            if ntwk2.has_edge(edge[0], edge[1]):
                dict2 = ntwk2.has_edge(edge[0], edge[1])
                dict1 = ntwk1.edge[edge[0]][edge[1]]
                data = add_dicts_by_key(dict1, dict2, subtract=True)
                diff_ntwk.add_edge(edge[0], edge[1], data)

    # Writes the networks and returns the name
    name = str(name2) + '-' + str(name1)
    
    if keep_only_common_edges == True:
        name = name + '_common_edges'
    elif keep_only_common_edges == False:
        name = name + '_uncommon'        
                
    network_name = 'difference_' + name
   
    nx.write_gpickle(diff_ntwk, op.abspath(network_name + '.pck'))
    iflogger.info('Saving difference graph as {out}'.format(out=op.abspath(network_name + '.pck')))
    diff_ntwk = fix_keys_for_gexf(diff_ntwk)
    
    nx.write_gexf(diff_ntwk, op.abspath(network_name + '.gexf'))
    iflogger.info('Saving difference graph as {out}'.format(out=op.abspath(network_name + '.gexf')))
    diff_array = nx.to_numpy_matrix(diff_ntwk)
    diff_array = np.array(diff_array)
    diff_dict = {}
    diff_dict[network_name] = diff_array
    sio.savemat(op.abspath(network_name + '.mat'), diff_dict)
    iflogger.info('Saving difference graph as {out}'.format(out=op.abspath(network_name + '.mat')))
    return network_name


class NetworkXMetricsInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Input network')
    out_k_core = File('k_core', usedefault=True, desc='Computed k-core network stored as a NetworkX pickle.')
    out_k_shell = File('k_shell', usedefault=True, desc='Computed k-shell network stored as a NetworkX pickle.')
    out_k_crust = File('k_crust', usedefault=True, desc='Computed k-crust network stored as a NetworkX pickle.')
    treat_as_weighted_graph = traits.Bool(True, usedefault=True, desc='Some network metrics can be calculated while considering only a binarized version of the graph')
    compute_clique_related_measures = traits.Bool(False, usedefault=True, desc='Computing clique-related measures (e.g. node clique number) can be very time consuming')
    out_global_metrics_matlab = File(genfile=True, desc='Output node metrics in MATLAB .mat format')
    out_node_metrics_matlab = File(genfile=True, desc='Output node metrics in MATLAB .mat format')
    out_edge_metrics_matlab = File(genfile=True, desc='Output edge metrics in MATLAB .mat format')
    out_pickled_extra_measures = File('extra_measures', usedefault=True, desc='Network measures for group 1 that return dictionaries stored as a Pickle.')

class NetworkXMetricsOutputSpec(TraitedSpec):
    gpickled_network_files = OutputMultiPath(File(desc='Output gpickled network files'))
    matlab_matrix_files = OutputMultiPath(File(desc='Output network metrics in MATLAB .mat format'))
    global_measures_matlab = File(desc='Output global metrics in MATLAB .mat format')
    node_measures_matlab = File(desc='Output node metrics in MATLAB .mat format')
    edge_measures_matlab = File(desc='Output edge metrics in MATLAB .mat format')
    node_measure_networks = OutputMultiPath(File(desc='Output gpickled network files for all node-based measures'))
    edge_measure_networks = OutputMultiPath(File(desc='Output gpickled network files for all edge-based measures'))
    k_networks = OutputMultiPath(File(desc='Output gpickled network files for the k-core, k-shell, and k-crust networks'))
    k_core = File(desc='Computed k-core network stored as a NetworkX pickle.')
    k_shell = File(desc='Computed k-shell network stored as a NetworkX pickle.')
    k_crust = File(desc='Computed k-crust network stored as a NetworkX pickle.')
    pickled_extra_measures = File(desc='Network measures for the group that return dictionaries, stored as a Pickle.')
    matlab_dict_measures = OutputMultiPath(File(desc='Network measures for the group that return dictionaries, stored as matlab matrices.'))

class NetworkXMetrics(BaseInterface):
    """
    Calculates and outputs NetworkX-based measures for an input network

    Example
    -------

    >>> import nipype.interfaces.cmtk as cmtk
    >>> nxmetrics = cmtk.NetworkXMetrics()
    >>> nxmetrics.inputs.in_file = 'subj1.pck'
    >>> nxmetrics.run()                 # doctest: +SKIP
    """
    input_spec = NetworkXMetricsInputSpec
    output_spec = NetworkXMetricsOutputSpec

    def _run_interface(self, runtime):
        global gpickled, nodentwks, edgentwks, kntwks, matlab
        gpickled = list()
        nodentwks = list()
        edgentwks = list()
        kntwks = list()
        matlab = list()
        ntwk = nx.read_gpickle(self.inputs.in_file)

        # Each block computes, writes, and saves a measure
        # The names are then added to the output .pck file list
        # In the case of the degeneracy networks, they are given specified output names

        calculate_cliques = self.inputs.compute_clique_related_measures
        weighted = self.inputs.treat_as_weighted_graph

        global_measures = compute_singlevalued_measures(ntwk, weighted, calculate_cliques)
        if isdefined(self.inputs.out_global_metrics_matlab):
            global_out_file = op.abspath(self.inputs.out_global_metrics_matlab)
        else:
            global_out_file = op.abspath(self._gen_outfilename('globalmetrics', 'mat'))
        sio.savemat(global_out_file, global_measures, oned_as='column')
        matlab.append(global_out_file)

        node_measures = compute_node_measures(ntwk, calculate_cliques)
        for key in node_measures.keys():
            newntwk = add_node_data(node_measures[key], ntwk)
            out_file = op.abspath(self._gen_outfilename(key, 'pck'))
            nx.write_gpickle(newntwk, out_file)
            nodentwks.append(out_file)
        if isdefined(self.inputs.out_node_metrics_matlab):
            node_out_file = op.abspath(self.inputs.out_node_metrics_matlab)
        else:
            node_out_file = op.abspath(self._gen_outfilename('nodemetrics', 'mat'))
        sio.savemat(node_out_file, node_measures, oned_as='column')
        matlab.append(node_out_file)
        gpickled.extend(nodentwks)

        edge_measures = compute_edge_measures(ntwk)
        for key in edge_measures.keys():
            newntwk = add_edge_data(edge_measures[key], ntwk)
            out_file = op.abspath(self._gen_outfilename(key, 'pck'))
            nx.write_gpickle(newntwk, out_file)
            edgentwks.append(out_file)
        if isdefined(self.inputs.out_edge_metrics_matlab):
            edge_out_file = op.abspath(self.inputs.out_edge_metrics_matlab)
        else:
            edge_out_file = op.abspath(self._gen_outfilename('edgemetrics', 'mat'))
        sio.savemat(edge_out_file, edge_measures, oned_as='column')
        matlab.append(edge_out_file)
        gpickled.extend(edgentwks)

        ntwk_measures = compute_network_measures(ntwk)
        for key in ntwk_measures.keys():
            if key == 'k_core':
                out_file = op.abspath(self._gen_outfilename(self.inputs.out_k_core, 'pck'))
            if key == 'k_shell':
                out_file = op.abspath(self._gen_outfilename(self.inputs.out_k_shell, 'pck'))
            if key == 'k_crust':
                out_file = op.abspath(self._gen_outfilename(self.inputs.out_k_crust, 'pck'))
            nx.write_gpickle(ntwk_measures[key], out_file)
            kntwks.append(out_file)
        gpickled.extend(kntwks)

        out_pickled_extra_measures = op.abspath(self._gen_outfilename(self.inputs.out_pickled_extra_measures, 'pck'))
        dict_measures = compute_dict_measures(ntwk)
        iflogger.info('Saving extra measure file to {path} in Pickle format'.format(path=op.abspath(out_pickled_extra_measures)))
        file = open(out_pickled_extra_measures, 'w')
        pickle.dump(dict_measures, file)
        file.close()

        iflogger.info('Saving MATLAB measures as {m}'.format(m=matlab))

        # Loops through the measures which return a dictionary,
        # converts the keys and values to a Numpy array,
        # stacks them together, and saves them in a MATLAB .mat file via Scipy
        global dicts
        dicts = list()
        for idx, key in enumerate(dict_measures.keys()):
            for idxd, keyd in enumerate(dict_measures[key].keys()):
                if idxd == 0:
                    nparraykeys = np.array(keyd)
                    nparrayvalues = np.array(dict_measures[key][keyd])
                else:
                    nparraykeys = np.append(nparraykeys, np.array(keyd))
                    values = np.array(dict_measures[key][keyd])
                    nparrayvalues = np.append(nparrayvalues, values)
            nparray = np.vstack((nparraykeys, nparrayvalues))
            out_file = op.abspath(self._gen_outfilename(key, 'mat'))
            npdict = {}
            npdict[key] = nparray
            sio.savemat(out_file, npdict, oned_as='column')
            dicts.append(out_file)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs["k_core"] = op.abspath(self._gen_outfilename(self.inputs.out_k_core, 'pck'))
        outputs["k_shell"] = op.abspath(self._gen_outfilename(self.inputs.out_k_shell, 'pck'))
        outputs["k_crust"] = op.abspath(self._gen_outfilename(self.inputs.out_k_crust, 'pck'))
        outputs["gpickled_network_files"] = gpickled
        outputs["k_networks"] = kntwks
        outputs["node_measure_networks"] = nodentwks
        outputs["edge_measure_networks"] = edgentwks
        outputs["matlab_dict_measures"] = dicts
        outputs["global_measures_matlab"] = op.abspath(self._gen_outfilename('globalmetrics', 'mat'))
        outputs["node_measures_matlab"] = op.abspath(self._gen_outfilename('nodemetrics', 'mat'))
        outputs["edge_measures_matlab"] = op.abspath(self._gen_outfilename('edgemetrics', 'mat'))
        outputs["matlab_matrix_files"] = [outputs["global_measures_matlab"], outputs["node_measures_matlab"], outputs["edge_measures_matlab"]]
        outputs["pickled_extra_measures"] = op.abspath(self._gen_outfilename(self.inputs.out_pickled_extra_measures, 'pck'))
        return outputs

    def _gen_outfilename(self, name, ext):
        return name + '.' + ext

class AverageNetworksInputSpec(BaseInterfaceInputSpec):
    in_files = InputMultiPath(File(exists=True), mandatory=True, desc='Networks for a group of subjects')
    resolution_network_file = File(exists=True, desc='Parcellation files from Connectome Mapping Toolkit. This is not necessary' \
                                ', but if included, the interface will output the statistical maps as networkx graphs.')
    group_id = traits.Str('group1', usedefault=True, desc='ID for group')
    out_gpickled_groupavg = File(desc='Average network saved as a NetworkX .pck')
    out_gexf_groupavg = File(desc='Average network saved as a .gexf file')

class AverageNetworksOutputSpec(TraitedSpec):
    gpickled_groupavg = File(desc='Average network saved as a NetworkX .pck')
    gexf_groupavg = File(desc='Average network saved as a .gexf file')
    matlab_groupavgs = OutputMultiPath(File(desc='Average network saved as a .gexf file'))

class AverageNetworks(BaseInterface):
    """
    Calculates and outputs the average network given a set of input NetworkX gpickle files

    This interface will only keep an edge in the averaged network if that edge is present in
    at least half of the input networks.

    Example
    -------

    >>> import nipype.interfaces.cmtk as cmtk
    >>> avg = cmtk.AverageNetworks()
    >>> avg.inputs.in_files = ['subj1.pck', 'subj2.pck']
    >>> avg.run()                 # doctest: +SKIP

    """
    input_spec = AverageNetworksInputSpec
    output_spec = AverageNetworksOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.resolution_network_file):
            ntwk_res_file = self.inputs.resolution_network_file
        else:
            ntwk_res_file = self.inputs.in_files[0]

        global matlab_network_list
        network_name, matlab_network_list = average_networks(self.inputs.in_files, ntwk_res_file, self.inputs.group_id)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        if not isdefined(self.inputs.out_gpickled_groupavg):
            outputs["gpickled_groupavg"] = op.abspath(self._gen_outfilename(self.inputs.group_id + '_average', 'pck'))
        else:
            outputs["gpickled_groupavg"] = op.abspath(self.inputs.out_gpickled_groupavg)

        if not isdefined(self.inputs.out_gexf_groupavg):
            outputs["gexf_groupavg"] = op.abspath(self._gen_outfilename(self.inputs.group_id + '_average', 'gexf'))
        else:
            outputs["gexf_groupavg"] = op.abspath(self.inputs.out_gexf_groupavg)

        outputs["matlab_groupavgs"] = matlab_network_list
        return outputs

    def _gen_outfilename(self, name, ext):
        return name + '.' + ext

class DifferenceGraphInputSpec(BaseInterfaceInputSpec):
    in_file1 = File(exists=True, mandatory=True, desc='Network 1 for the equation: difference graph = in_file2 - in_file1')
    in_file2 = File(exists=True, mandatory=True, desc='Network 2 for the equation: difference graph = in_file2 - in_file1')
    keep_only_common_edges = traits.Bool(desc='Only the edges common to both networks are kept in the difference graph.' \
                                 'If False, only uncommon edges are kept. If undefined, all edges are considered')
    resolution_network_file = File(exists=True, desc='A network which defines where to place the nodes for the difference graph' \
                                'If this is not provided, the interface will take node positions from in_file2.')
    out_gpickled_difference = File(desc='Difference network saved as a NetworkX .pck')
    out_gexf_difference = File(desc='Difference network saved as a .gexf file')
    out_matlab_difference = File(desc='Difference network saved as a .mat file')

class DifferenceGraphOutputSpec(TraitedSpec):
    gpickled_difference_graph = File(desc='Difference network saved as a NetworkX .pck')
    gexf_difference_graph = File(desc='Difference network saved as a .gexf file')
    matlab_difference_graph = File(desc='Difference network saved as a MATLAB .mat file')

class DifferenceGraph(BaseInterface):
    """
    Calculates and outputs the difference network given two input NetworkX gpickle files
    
    * difference graph = in_file2 - in_file1
    
    By default this interface will perform basic subtraction (if keep_only_common_edges is left undefined). 
    The user can also specify whether to retain only the common (keep_only_common_edges = True) or 
    uncommon (keep_only_common_edges = False) edges between the two graphs.
    
    Node positions and data can be input using a network resolution file. If one is not specified, they 
    will be pulled from in_file2.

    Example
    -------

    >>> import nipype.interfaces.cmtk as cmtk
    >>> diff = cmtk.DifferenceGraph()
    >>> diff.inputs.in_file1 = 'subj1.pck'
    >>> diff.inputs.in_file2 = 'subj2.pck'
    >>> diff.run()                 # doctest: +SKIP

    """
    input_spec = DifferenceGraphInputSpec
    output_spec = DifferenceGraphOutputSpec

    def _run_interface(self, runtime):
        if isdefined(self.inputs.resolution_network_file):
            ntwk_res_file = self.inputs.resolution_network_file
        else:
            ntwk_res_file = self.inputs.in_file2

        if isdefined(self.inputs.keep_only_common_edges):
            network_name = difference_graph(self.inputs.in_file1, self.inputs.in_file2, ntwk_res_file, self.inputs.keep_only_common_edges)
        else:
            network_name = difference_graph(self.inputs.in_file1, self.inputs.in_file2, ntwk_res_file)
        return runtime

    def _list_outputs(self):
        outputs = self.output_spec().get()
        _, name1, _  = split_filename(self.inputs.in_file1)
        _, name2, _  = split_filename(self.inputs.in_file2)

        name = str(name2) + '-' + str(name1)

        if isdefined(self.inputs.keep_only_common_edges):
            if self.inputs.keep_only_common_edges == True:
                name = name + '_common_edges'
            elif self.inputs.keep_only_common_edges == False:
                name = name + '_uncommon'        
        
        if not isdefined(self.inputs.out_gpickled_difference):
            outputs["gpickled_difference_graph"] = op.abspath(self._gen_outfilename('difference_' + name, 'pck'))
        else:
            outputs["gpickled_difference_graph"] = op.abspath(self.inputs.out_gpickled_difference)

        if not isdefined(self.inputs.out_gexf_difference):
            outputs["gexf_difference_graph"] = op.abspath(self._gen_outfilename('difference_' + name, 'gexf'))
        else:
            outputs["gexf_difference_graph"] = op.abspath(self.inputs.out_gexf_difference)

        if not isdefined(self.inputs.out_gexf_difference):
            outputs["matlab_difference_graph"] = op.abspath(self._gen_outfilename('difference_' + name, 'mat'))
        else:
            outputs["matlab_difference_graph"] = op.abspath(self.inputs.out_gexf_difference)
            
        return outputs

    def _gen_outfilename(self, name, ext):
        return name + '.' + ext



def ntwk_to_nifti_image(in_file, weight_key='value'):
	import numpy as np
	import nibabel as nb
	import networkx as nx
	import os, os.path as op
	from nipype.utils.filemanip import split_filename	
	path, name, ext = split_filename(in_file)
	out_file = op.abspath(name + '.nii')
	try:
		ntwk = nx.read_graphml(in_file)
	except:
		ntwk = nx.read_gpickle(in_file)
		
	edge_array = np.asarray(nx.to_numpy_matrix(ntwk))
	header = nb.Nifti1Header()
	affine = np.eye(4)
	out_image = nb.Nifti1Image(data=edge_array, affine=affine, header=header)
	nb.save(out_image, out_file)
	return out_file


class Network2NiftiImageInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Network to be converted')
    weight_key = traits.Str('value', usedefault=True, desc='The edge key for the connectivity values (default: "value")')
    out_file = File(desc='Difference network saved as a .mat file')

class Network2NiftiImageOutputSpec(TraitedSpec):
    out_file = File(desc="Nifti image for the input network's connectivity matrix")

class Network2NiftiImage(BaseInterface):
	"""
	Converts a NetworkX graph in either gpickle or gexf format to a Nifti image

	Example
	-------

	>>> import nipype.interfaces.cmtk as cmtk
	>>> ntwk2nii = cmtk.Network2NiftiImage()
	>>> ntwk2nii.inputs.in_file = 'subj1.pck'
	>>> ntwk2nii.run()                 # doctest: +SKIP

	"""
	input_spec = Network2NiftiImageInputSpec
	output_spec = Network2NiftiImageOutputSpec

	def _run_interface(self, runtime):
		network_name = ntwk_to_nifti_image(self.inputs.in_file, self.inputs.weight_key)
		iflogger.info('Saving connectivity matrix to {path} as a Nifti image'.format(path=op.abspath(network_name)))
		return runtime

	def _list_outputs(self):
		outputs = self.output_spec().get()
		path, name, ext = split_filename(self.inputs.in_file)
		outputs["out_file"] = op.abspath(name + '.nii')
		return outputs


def common_edges(in_file, filter_file, ntwk_res_file):
	"""
	Filters the edges of in_file using the edges of filter_file and keeps the node positions of ntwk_res_file.

	"""
	import networkx as nx
	import os.path as op
	from nipype.utils.filemanip import split_filename
	iflogger.info("Creating common edge network: {in_f} with edges in {f}".format(in_f=in_file, f=filter_file))

	_, name, _ = split_filename(in_file)
	_, filter_name, _ = split_filename(filter_file)
	ntwk = nx.read_gpickle(in_file)
	iflogger.info('File {s} has {n} edges'.format(s=name, n=ntwk.number_of_edges()))
	filter_ntwk = nx.read_gpickle(filter_file)
	iflogger.info('File {s} has {n} edges'.format(s=filter_name, n=filter_ntwk.number_of_edges()))

	ntwk_res_file = nx.read_gpickle(ntwk_res_file)
	iflogger.info("{n} Nodes found in network resolution file".format(n=ntwk_res_file.number_of_nodes()))

	common_ntwk = ntwk_res_file.copy()
	common_ntwk = remove_all_edges(common_ntwk)
	edges1 = ntwk.edges_iter()
	filter_edges = filter_ntwk.edges_iter()
	for edge in filter_edges:
		# Retain only edges in both networks and calculate the difference for each key at those edges
		if ntwk.has_edge(edge[0], edge[1]):
			dict_in = ntwk.edge[edge[0]][edge[1]]
			common_ntwk.add_edge(edge[0], edge[1], dict_in)

	# Writes the networks and returns the name
	network_name = str(name) + '-edges_filtered_by-' + str(filter_name)

	nx.write_gpickle(common_ntwk, op.abspath(network_name + '.pck'))
	iflogger.info('Saving common edge graph as {out}'.format(out=op.abspath(network_name + '.pck')))
	common_ntwk = fix_keys_for_gexf(common_ntwk)

	nx.write_gexf(common_ntwk, op.abspath(network_name + '.gexf'))
	iflogger.info('Saving common edge graph as {out}'.format(out=op.abspath(network_name + '.gexf')))
	common_array = nx.to_numpy_matrix(common_ntwk)
	common_array = np.array(common_array)
	common_dict = {}
	common_dict[network_name] = common_array
	sio.savemat(op.abspath(network_name + '.mat'), common_dict)
	iflogger.info('Saving common edge graph as {out}'.format(out=op.abspath(network_name + '.mat')))
	return network_name

class CommonEdgesInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Network to filter')
    filter_file = File(exists=True, mandatory=True, desc='Only edges in this network, with weights from in_file, will be returned')
    resolution_network_file = File(exists=True, desc='A network which defines where to place the nodes for the filtered graph' \
                                'If this is not provided, the interface will take node positions from in_file.')
    out_gpickled_common_edges = File(desc='Filtered network saved as a NetworkX .pck')
    out_gexf_common_edges = File(desc='Filtered network saved as a .gexf file')
    out_matlab_common_edges = File(desc='Filtered network saved as a .mat file')

class CommonEdgesOutputSpec(TraitedSpec):
    gpickled_common_edges = File(desc='Difference network saved as a NetworkX .pck')
    gexf_common_edges = File(desc='Difference network saved as a .gexf file')
    matlab_common_edges = File(desc='Difference network saved as a MATLAB .mat file')

class CommonEdges(BaseInterface):
	"""
	Filters a graph's edges using another graph given two input NetworkX gpickle files

	Node positions and data can be input using a network resolution file. If one is not specified, they 
	will be pulled from in_file2.

	Example
	-------

	>>> import nipype.interfaces.cmtk as cmtk
	>>> diff = cmtk.CommonEdges()
	>>> diff.inputs.in_file = 'subj1.pck'
	>>> diff.inputs.filter_file = 'subj2.pck'
	>>> diff.run()                 # doctest: +SKIP

	"""
	input_spec = CommonEdgesInputSpec
	output_spec = CommonEdgesOutputSpec

	def _run_interface(self, runtime):
		if isdefined(self.inputs.resolution_network_file):
			ntwk_res_file = self.inputs.resolution_network_file
		else:
			ntwk_res_file = self.inputs.in_file
		
		network_name = common_edges(self.inputs.in_file, self.inputs.filter_file, ntwk_res_file)
		return runtime

	def _list_outputs(self):
		outputs = self.output_spec().get()
		_, name, _  = split_filename(self.inputs.in_file)
		_, filter_name, _  = split_filename(self.inputs.filter_file)

		name = str(name) + '-edges_filtered_by-' + str(filter_name)
		
		if not isdefined(self.inputs.out_gpickled_common_edges):
			outputs["gpickled_common_edges"] = op.abspath(self._gen_outfilename('common_' + name, 'pck'))
		else:
			outputs["gpickled_common_edges"] = op.abspath(self.inputs.out_gpickled_common_edges)

		if not isdefined(self.inputs.out_gexf_common_edges):
			outputs["gexf_common_edges"] = op.abspath(self._gen_outfilename('common_' + name, 'gexf'))
		else:
			outputs["gexf_common_edges"] = op.abspath(self.inputs.out_gexf_common_edges)

		if not isdefined(self.inputs.out_matlab_common_edges):
			outputs["matlab_common_edges"] = op.abspath(self._gen_outfilename('common_' + name, 'mat'))
		else:
			outputs["matlab_common_edges"] = op.abspath(self.inputs.out_matlab_common_edges)
			
		return outputs

	def _gen_outfilename(self, name, ext):
		return name + '.' + ext
		
		
def threshold_edges(in_file, edge_key, threshold=0.05, above=False):
	"""
	Filters the edges of in_file using the edges of filter_file and keeps the node positions of ntwk_res_file.

	"""
	import networkx as nx
	import os.path as op
	from nipype.utils.filemanip import split_filename
	iflogger.info("Thresholding edges in network: {in_f} with threshold {t}".format(in_f=in_file, t=threshold))

	_, name, _ = split_filename(in_file)
	ntwk = nx.read_gpickle(in_file)
	iflogger.info('File {s} has {n} edges'.format(s=name, n=ntwk.number_of_edges()))

	thresholded_ntwk = ntwk.copy()
	thresholded_ntwk = remove_all_edges(thresholded_ntwk)
	edges1 = ntwk.edges_iter()
	for edge in edges1:
		# Retain only edges that are above or below the threshold
		value = ntwk.edge[edge[0]][edge[1]][edge_key]
		if above and value > threshold:
			dict_in = ntwk.edge[edge[0]][edge[1]]
			thresholded_ntwk.add_edge(edge[0], edge[1], dict_in)
		elif above == False and value <= threshold:
			dict_in = ntwk.edge[edge[0]][edge[1]]
			thresholded_ntwk.add_edge(edge[0], edge[1], dict_in)

	# Writes the networks and returns the name
	network_name = str(name) + '-thresholded'

	nx.write_gpickle(thresholded_ntwk, op.abspath(network_name + '.pck'))
	iflogger.info('Saving thresholded graph as {out}'.format(out=op.abspath(network_name + '.pck')))
	thresholded_ntwk = fix_keys_for_gexf(thresholded_ntwk)

	nx.write_gexf(thresholded_ntwk, op.abspath(network_name + '.gexf'))
	iflogger.info('Saving thresholded graph as {out}'.format(out=op.abspath(network_name + '.gexf')))
	thresholded_array = nx.to_numpy_matrix(thresholded_ntwk)
	thresholded_array = np.array(thresholded_array)
	thresholded_dict = {}
	thresholded_dict[network_name] = thresholded_array
	sio.savemat(op.abspath(network_name + '.mat'), thresholded_dict)
	iflogger.info('Saving thresholded graph as {out}'.format(out=op.abspath(network_name + '.mat')))
	return network_name

class ThresholdNetworkInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='Network to filter')
    above_threshold = traits.Bool(False, usedefault=True, desc='Only connectivity values greater than or equal to the threshold will be used')
    weight_threshold = traits.Float(0.05, usedefault=True, desc='Connectivity weight threshold (default 0.05, for NBS graph analysis)')
    edge_key = traits.Str('weight', usedefault=True, desc='Connectivity edge key to threshold')
    out_gpickled_network = File(desc='Filtered network saved as a NetworkX .pck')
    out_gexf_network = File(desc='Filtered network saved as a .gexf file')
    out_matlab_network = File(desc='Filtered network saved as a .mat file')

class ThresholdNetworkOutputSpec(TraitedSpec):
    gpickled_network = File(desc='Thresholded network saved as a NetworkX .pck')
    gexf_network = File(desc='Thresholded network saved as a .gexf file')
    matlab_network = File(desc='Thresholded network saved as a MATLAB .mat file')

class ThresholdNetwork(BaseInterface):
	"""
	Filters a graph's edges given a threshold value

	Example
	-------

	>>> import nipype.interfaces.cmtk as cmtk
	>>> thresh = cmtk.ThresholdNetwork()
	>>> thresh.inputs.in_file = 'subj1.pck'
	>>> thresh.run()                 # doctest: +SKIP

	"""
	input_spec = ThresholdNetworkInputSpec
	output_spec = ThresholdNetworkOutputSpec

	def _run_interface(self, runtime):	
		network_name = threshold_edges(self.inputs.in_file, self.inputs.edge_key, self.inputs.weight_threshold, self.inputs.above_threshold)
		return runtime

	def _list_outputs(self):
		outputs = self.output_spec().get()
		_, name, _  = split_filename(self.inputs.in_file)

		network_name = str(name) + '-thresholded'
		
		if not isdefined(self.inputs.out_gpickled_network):
			outputs["gpickled_network"] = op.abspath(self._gen_outfilename(network_name, 'pck'))
		else:
			outputs["gpickled_network"] = op.abspath(self.inputs.out_gpickled_network)

		if not isdefined(self.inputs.out_gexf_network):
			outputs["gexf_network"] = op.abspath(self._gen_outfilename(network_name, 'gexf'))
		else:
			outputs["gexf_network"] = op.abspath(self.inputs.out_gexf_common_edges)

		if not isdefined(self.inputs.out_matlab_network):
			outputs["matlab_network"] = op.abspath(self._gen_outfilename(network_name, 'mat'))
		else:
			outputs["matlab_network"] = op.abspath(self.inputs.out_matlab_network)
			
		return outputs

	def _gen_outfilename(self, name, ext):
		return name + '.' + ext

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
