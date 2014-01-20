# AUTO-GENERATED by tools/checkspecs.py - DO NOT EDIT
from nipype.testing import assert_equal
from nipype.interfaces.afni.preprocess import Volreg
def test_Volreg_inputs():
    input_map = dict(oned_file=dict(name_source='in_file',
    argstr='-1Dfile %s',
    name_template='%s.1D',
    keep_extension=True,
    ),
    ignore_exception=dict(nohash=True,
    usedefault=True,
    ),
    verbose=dict(argstr='-verbose',
    ),
    timeshift=dict(argstr='-tshift 0',
    ),
    basefile=dict(position=-6,
    argstr='-base %s',
    ),
    args=dict(argstr='%s',
    ),
    outputtype=dict(),
    zpad=dict(position=-5,
    argstr='-zpad %d',
    ),
    terminal_output=dict(mandatory=True,
    nohash=True,
    ),
    md1d_file=dict(name_source='in_file',
    keep_extension=True,
    position=-4,
    name_template='%s_md.1D',
    argstr='-maxdisp1D %s',
    ),
    in_file=dict(copyfile=False,
    mandatory=True,
    position=-1,
    argstr='%s',
    ),
    copyorigin=dict(argstr='-twodup',
    ),
    environ=dict(nohash=True,
    usedefault=True,
    ),
    out_file=dict(name_source='in_file',
    name_template='%s_volreg',
    argstr='-prefix %s',
    ),
    )
    inputs = Volreg.input_spec()

    for key, metadata in input_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(inputs.traits()[key], metakey), value
def test_Volreg_outputs():
    output_map = dict(oned_file=dict(),
    md1d_file=dict(),
    out_file=dict(),
    )
    outputs = Volreg.output_spec()

    for key, metadata in output_map.items():
        for metakey, value in metadata.items():
            yield assert_equal, getattr(outputs.traits()[key], metakey), value
