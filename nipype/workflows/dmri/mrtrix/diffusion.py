import nipype.interfaces.utility as util     # utility
import nipype.pipeline.engine as pe          # pypeline engine
import nipype.interfaces.fsl as fsl
import nipype.interfaces.mrtrix as mrtrix

def create_mrtrix_dti_pipeline(name="dtiproc", tractography_type = 'probabilistic'):
    """Creates a pipeline that does the same diffusion processing as in the
    :doc:`../../users/examples/dmri_mrtrix_dti` example script. Given a diffusion-weighted image,
    b-values, and b-vectors, the workflow will return the tractography
    computed from spherical deconvolution and probabilistic streamline tractography

    Example
    -------

    >>> dti = create_mrtrix_dti_pipeline("mrtrix_dti")
    >>> dti.inputs.inputnode.dwi = 'data.nii'
    >>> dti.inputs.inputnode.bvals = 'bvals'
    >>> dti.inputs.inputnode.bvecs = 'bvecs'
    >>> dti.run()                  # doctest: +SKIP

    Inputs::

        inputnode.dwi
        inputnode.bvecs
        inputnode.bvals

    Outputs::

        outputnode.fa
        outputnode.tdi
        outputnode.tracts_tck
        outputnode.tracts_trk
        outputnode.csdeconv

    """

    inputnode = pe.Node(interface = util.IdentityInterface(fields=["dwi",
                                                                   "bvecs",
                                                                   "bvals"]),
                        name="inputnode")

    bet = pe.Node(interface=fsl.BET(), name="bet")
    bet.inputs.mask = True

    fsl2mrtrix = pe.Node(interface=mrtrix.FSL2MRTrix(),name='fsl2mrtrix')
    fsl2mrtrix.inputs.invert_y = True

    dwi2tensor = pe.Node(interface=mrtrix.DWI2Tensor(),name='dwi2tensor')

    tensor2vector = pe.Node(interface=mrtrix.Tensor2Vector(),
                            name='tensor2vector')
    tensor2adc = pe.Node(interface=mrtrix.Tensor2ApparentDiffusion(),
                         name='tensor2adc')
    tensor2fa = pe.Node(interface=mrtrix.Tensor2FractionalAnisotropy(),
                        name='tensor2fa')

    erode_mask_firstpass = pe.Node(interface=mrtrix.Erode(),
                                   name='erode_mask_firstpass')
    erode_mask_secondpass = pe.Node(interface=mrtrix.Erode(),
                                    name='erode_mask_secondpass')

    threshold_b0 = pe.Node(interface=mrtrix.Threshold(),name='threshold_b0')

    threshold_FA = pe.Node(interface=mrtrix.Threshold(),name='threshold_FA')
    threshold_FA.inputs.absolute_threshold_value = 0.7

    threshold_wmmask = pe.Node(interface=mrtrix.Threshold(),
                               name='threshold_wmmask')
    threshold_wmmask.inputs.absolute_threshold_value = 0.4

    MRmultiply = pe.Node(interface=mrtrix.MRMultiply(),name='MRmultiply')
    MRmult_merge = pe.Node(interface=util.Merge(2), name='MRmultiply_merge')

    median3d = pe.Node(interface=mrtrix.MedianFilter3D(),name='median3D')

    MRconvert = pe.Node(interface=mrtrix.MRConvert(),name='MRconvert')
    MRconvert.inputs.extract_at_axis = 3
    MRconvert.inputs.extract_at_coordinate = [0]

    csdeconv = pe.Node(interface=mrtrix.ConstrainedSphericalDeconvolution(),
                       name='csdeconv')

    gen_WM_mask = pe.Node(interface=mrtrix.GenerateWhiteMatterMask(),
                          name='gen_WM_mask')

    estimateresponse = pe.Node(interface=mrtrix.EstimateResponseForSH(),
                               name='estimateresponse')

    if tractography_type == 'probabilistic':
        CSDstreamtrack = pe.Node(interface=mrtrix.ProbabilisticSphericallyDeconvolutedStreamlineTrack(),
                                 name='CSDstreamtrack')
    else:
        CSDstreamtrack = pe.Node(interface=mrtrix.SphericallyDeconvolutedStreamlineTrack(),
                                 name='CSDstreamtrack')
    CSDstreamtrack.inputs.desired_number_of_tracks = 15000

    tracks2prob = pe.Node(interface=mrtrix.Tracks2Prob(),name='tracks2prob')
    tracks2prob.inputs.colour = True
    tck2trk = pe.Node(interface=mrtrix.MRTrix2TrackVis(),name='tck2trk')

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    workflow.connect([(inputnode, fsl2mrtrix, [("bvecs", "bvec_file"),
                                                    ("bvals", "bval_file")])])
    workflow.connect([(inputnode, dwi2tensor,[("dwi","in_file")])])
    workflow.connect([(fsl2mrtrix, dwi2tensor,[("encoding_file","encoding_file")])])

    workflow.connect([(dwi2tensor, tensor2vector,[['tensor','in_file']]),
                           (dwi2tensor, tensor2adc,[['tensor','in_file']]),
                           (dwi2tensor, tensor2fa,[['tensor','in_file']]),
                          ])

    workflow.connect([(inputnode, tensor_mode, [("bvecs", "bvecs"),
                                                    ("bvals", "bvals")])])
    workflow.connect([(inputnode, tensor_mode,[("dwi","in_file")])])


    workflow.connect([(inputnode, MRconvert,[("dwi","in_file")])])
    workflow.connect([(MRconvert, threshold_b0,[("converted","in_file")])])
    workflow.connect([(threshold_b0, median3d,[("out_file","in_file")])])
    workflow.connect([(median3d, erode_mask_firstpass,[("out_file","in_file")])])
    workflow.connect([(erode_mask_firstpass, erode_mask_secondpass,[("out_file","in_file")])])

    workflow.connect([(tensor2fa, MRmult_merge,[("FA","in1")])])
    workflow.connect([(erode_mask_secondpass, MRmult_merge,[("out_file","in2")])])
    workflow.connect([(MRmult_merge, MRmultiply,[("out","in_files")])])
    workflow.connect([(MRmultiply, threshold_FA,[("out_file","in_file")])])
    workflow.connect([(threshold_FA, estimateresponse,[("out_file","mask_image")])])

    workflow.connect([(inputnode, bet,[("dwi","in_file")])])
    workflow.connect([(inputnode, gen_WM_mask,[("dwi","in_file")])])
    workflow.connect([(bet, gen_WM_mask,[("mask_file","binary_mask")])])
    workflow.connect([(fsl2mrtrix, gen_WM_mask,[("encoding_file","encoding_file")])])

    workflow.connect([(inputnode, estimateresponse,[("dwi","in_file")])])
    workflow.connect([(fsl2mrtrix, estimateresponse,[("encoding_file","encoding_file")])])

    workflow.connect([(inputnode, csdeconv,[("dwi","in_file")])])
    workflow.connect([(gen_WM_mask, csdeconv,[("WMprobabilitymap","mask_image")])])
    workflow.connect([(estimateresponse, csdeconv,[("response","response_file")])])
    workflow.connect([(fsl2mrtrix, csdeconv,[("encoding_file","encoding_file")])])

    workflow.connect([(gen_WM_mask, threshold_wmmask,[("WMprobabilitymap","in_file")])])
    workflow.connect([(threshold_wmmask, CSDstreamtrack,[("out_file","seed_file")])])
    workflow.connect([(csdeconv, CSDstreamtrack,[("spherical_harmonics_image","in_file")])])

    if tractography_type == 'probabilistic':
        workflow.connect([(CSDstreamtrack, tracks2prob,[("tracked","in_file")])])
        workflow.connect([(inputnode, tracks2prob,[("dwi","template_file")])])

    workflow.connect([(CSDstreamtrack, tck2trk,[("tracked","in_file")])])
    workflow.connect([(inputnode, tck2trk,[("dwi","image_file")])])

    output_fields = ["fa", "tracts_trk", "csdeconv", "tracts_tck"]
    if tractography_type == 'probabilistic':
        output_fields.append("tdi")
    outputnode = pe.Node(interface = util.IdentityInterface(fields=output_fields),
                                        name="outputnode")

    workflow.connect([(CSDstreamtrack, outputnode, [("tracked", "tracts_tck")]),
                      (csdeconv, outputnode, [("spherical_harmonics_image", "csdeconv")]),
                      (tensor2fa, outputnode, [("FA", "fa")]),
                      (tensor_mode, outputnode, [("out_file", "fa")]),
                      (tck2trk, outputnode, [("out_file", "tracts_trk")])

                      (tck2trk, outputnode, [("out_file", "tracts_trk")])
                      ])
    if tractography_type == 'probabilistic':
        workflow.connect([(tracks2prob, outputnode, [("tract_image", "tdi")])])

    return workflow
<<<<<<< HEAD
=======


def create_track_normalization_pipeline(name="normtracks"):
    """Creates a pipeline to normalize a set of tracks from a subject's
    diffusion space into a user-specified template space.

    Example
    -------

    >>> norm = create_track_normalization_pipeline("normtracks")
    >>> norm.inputs.inputnode.tracks = 'tracks.tck'
    >>> norm.inputs.inputnode.fa = 'fa.nii'
    >>> norm.inputs.inputnode.structural = 'struct.nii'
    >>> norm.inputs.inputnode.template = 't1.nii'
    >>> norm.run()                  # doctest: +SKIP

    Inputs::

        inputnode.tracks
        inputnode.fa
        inputnode.structural
        inputnode.template

    Outputs::

        outputnode.normalized_tracks

    """

    inputnode = pe.Node(interface = util.IdentityInterface(fields=["tracks",   
                                                                   "fa",
                                                                   "structural"]),
                        name="inputnode")

    def pull_prefix(in_files):
        from nipype.utils.filemanip import split_filename
        path, name, ext = split_filename(in_files[0])
        fixnaming = name[0:-2]
        return fixnaming

    gen_unit_warpfield_in_MNI = pe.Node(interface=mrtrix.GenerateUnitWarpField(), name='gen_unit_warpfield_in_MNI')
    #gen_unit_warp.inputs.in_file = '/path/to/spm8/toolbox/Seg/TPM.nii'

    norm_tracks = pe.Node(interface=mrtrix.NormalizeTracks(), name='norm_tracks')

    newsegment_T1 = pe.Node(interface=spm.NewSegment(), name='newsegment_T1')
    newsegment_T1.inputs.write_deformation_fields = [True, True]
    newsegment_T1.inputs.channel_info = (2, 60, (True, True))

    reslice_fa = pe.Node(interface=fs.MRIConvert(), name='reslice_fa')
    reslice_fa.inputs.out_type = 'nii'

    deform_fa = pe.Node(interface=spm.ApplyDeformations(), name='deform_fa')
    warp_MNI_to_T1 = pe.Node(interface=spm.ApplyDeformations(), name='warp_MNI_to_T1')
    warp_T1_to_FA = pe.Node(interface=spm.ApplyDeformations(), name='warp_T1_to_FA')

    extract_brain_from_T1 = pe.Node(interface=fsl.BET(), name='extract_brain_from_T1')
    extract_brain_from_T1.inputs.output_type = 'NIFTI'

    apply_deform_to_T1 = pe.Node(interface=spm.ApplyDeformations(), name='apply_deform_to_T1')
    applyxfm_T1_to_FA = pe.Node(interface=fsl.ApplyXfm(), name='applyxfm_T1_to_FA')
    applyxfm_T1_to_FA.inputs.apply_xfm = True
    applyxfm_T1_to_FA.inputs.no_resample = True
    applyxfm_T1_to_FA.inputs.output_type = 'NIFTI'

    invert_xfm = pe.Node(interface=fsl.ConvertXFM(), name='invert_xfm')
    invert_xfm.inputs.invert_xfm = True

    coreg_struct_fa = pe.Node(interface=fsl.FLIRT(), name='coreg_struct_fa')
    coreg_struct_fa.inputs.cost = 'normmi'
    coreg_struct_fa.inputs.dof = 12
    coreg_struct_fa.inputs.output_type = 'NIFTI'
    output_fields = ["normalized_tracks", "normalized_structural", "normalized_class_images", 
                    "normalized_FA", "dwi_to_struct_mat", "T1_in_dwi_space", "bias_corrected", "transform_images", "forward_deformation_field"]

    outputnode = pe.Node(interface = util.IdentityInterface(fields=output_fields),
                                        name="outputnode")

    workflow = pe.Workflow(name=name)
    workflow.base_output_dir=name

    workflow.connect([(inputnode, norm_tracks,[("tracks","in_file")])])
    workflow.connect([(inputnode, reslice_fa,[("fa","in_file")])])
    workflow.connect([(inputnode, reslice_fa,[("structural","reslice_like")])])
    workflow.connect([(inputnode, extract_brain_from_T1,[("structural","in_file")])])
    workflow.connect([(extract_brain_from_T1, coreg_struct_fa,[("out_file","reference")])])
    workflow.connect([(reslice_fa, coreg_struct_fa,[("out_file","in_file")])])
    workflow.connect([(coreg_struct_fa, outputnode,[("out_matrix_file","dwi_to_struct_mat")])])
    workflow.connect([(coreg_struct_fa, outputnode,[("out_file","T1_in_dwi_space")])])

    workflow.connect([(coreg_struct_fa, invert_xfm,[('out_matrix_file', 'in_file')])])
    workflow.connect([(invert_xfm, applyxfm_T1_to_FA,[('out_file', 'in_matrix_file')])])
    workflow.connect([(reslice_fa, applyxfm_T1_to_FA,[("out_file","reference")])])
    workflow.connect([(inputnode, applyxfm_T1_to_FA,[("structural","in_file")])])

    workflow.connect([(applyxfm_T1_to_FA, newsegment_T1,[("out_file","channel_files")])])

    workflow.connect([(inputnode, deform_fa,[("fa","in_files")])])
    workflow.connect([(newsegment_T1, deform_fa,[("forward_deformation_field","deformation_field")])])
    workflow.connect([(apply_deform_to_T1, deform_fa,[("out_files","reference_volume")])])
    workflow.connect([(deform_fa, outputnode,[("out_files","normalized_FA")])])

    workflow.connect([(gen_unit_warpfield_in_MNI, warp_MNI_to_T1,[('out_files', 'in_files')])])
    workflow.connect([(newsegment_T1, warp_MNI_to_T1,[("bias_corrected_images","inverse_volume")])])
    workflow.connect([(newsegment_T1, warp_MNI_to_T1,[('forward_deformation_field', 'inverse_deformation_field')])])
    workflow.connect([(warp_MNI_to_T1, norm_tracks,[('out_files', 'transform_images')])])
    workflow.connect([(warp_MNI_to_T1, norm_tracks, [(('out_files', pull_prefix), 'transform_image_prefix')])])

    workflow.connect([(norm_tracks, outputnode, [("out_file", "normalized_tracks")])])
    workflow.connect([(newsegment_T1, outputnode, [("normalized_class_images", "normalized_class_images")])])
    workflow.connect([(newsegment_T1, outputnode, [("forward_deformation_field", "forward_deformation_field")])])
    workflow.connect([(newsegment_T1, outputnode, [("bias_corrected_images", "bias_corrected")])])

    workflow.connect([(newsegment_T1, apply_deform_to_T1, [("forward_deformation_field", "deformation_field")])])
    workflow.connect([(newsegment_T1, apply_deform_to_T1, [("bias_corrected_images", "in_files")])])
    workflow.connect([(apply_deform_to_T1, outputnode, [("out_files", "normalized_structural")])])
    return workflow
>>>>>>> Updates Tracks2prob
