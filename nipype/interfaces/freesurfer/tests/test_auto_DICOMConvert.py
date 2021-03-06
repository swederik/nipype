# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.freesurfer.preprocess import DICOMConvert

def test_DICOMConvert_inputs():
    input_map = dict(args=dict(argstr='%s',
    ),
    base_output_dir=dict(mandatory=True,
    ),
    dicom_dir=dict(mandatory=True,
    ),
    dicom_info=dict(),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    file_mapping=dict(),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    ignore_single_slice=dict(requires=['dicom_info'],
    ),
    out_type=dict(usedefault=True,
    ),
    seq_list=dict(requires=['dicom_info'],
    ),
    subject_dir_template=dict(usedefault=True,
    ),
    subject_id=dict(),
    subjects_dir=dict(),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    )
    inputs = DICOMConvert.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

