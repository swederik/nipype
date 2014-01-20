# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.afni.preprocess import Bandpass
def test_Bandpass_inputs():
    input_map = dict(ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    localPV=dict(argstr='-localPV %f',
    ),
    orthogonalize_file=dict(argstr='-ort %s',
    ),
    nfft=dict(argstr='-nfft %d',
    ),
    tr=dict(argstr='-dt %f',
    ),
    despike=dict(argstr='-despike',
    ),
    args=dict(argstr='%s',
    ),
    no_detrend=dict(argstr='-nodetrend',
    ),
    outputtype=dict(),
    orthogonalize_dset=dict(argstr='-dsort %s',
    ),
    highpass=dict(position=-3,
    mandatory=True,
    argstr='%f',
    ),
    normalize=dict(argstr='-norm',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    notrans=dict(argstr='-notrans',
    ),
    in_file=dict(copyfile=False,
    mandatory=True,
    position=-1,
    argstr='%s',
    ),
    lowpass=dict(position=-2,
    mandatory=True,
    argstr='%f',
    ),
    blur=dict(argstr='-blur %f',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    automask=dict(argstr='-automask',
    ),
    mask=dict(position=2,
    argstr='-mask %s',
    ),
    out_file=dict(name_source='in_file',
    genfile=True,
    name_template='%s_bp',
    position=1,
    argstr='-prefix %s',
    ),
    )
    inputs = Bandpass.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_Bandpass_outputs():
    output_map = dict(out_file=dict(),
    )
    outputs = Bandpass.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
