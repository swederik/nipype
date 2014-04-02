# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.dipy.tensors import EstimateConductivity

def test_EstimateConductivity_inputs():
    input_map = dict(eigenvalue_scaling_factor=dict(units='NA',
    usedefault=True,
    ),
    in_file=dict(mandatory=True,
    ),
    lower_triangular_input=dict(usedefault=True,
    ),
    lower_triangular_output=dict(usedefault=True,
    ),
    out_filename=dict(genfile=True,
    ),
    sigma_white_matter=dict(units='NA',
    usedefault=True,
    ),
    use_outlier_correction=dict(usedefault=True,
    ),
    volume_normalized_mapping=dict(usedefault=True,
    ),
    )
    inputs = EstimateConductivity.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value

def test_EstimateConductivity_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = EstimateConductivity.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value

