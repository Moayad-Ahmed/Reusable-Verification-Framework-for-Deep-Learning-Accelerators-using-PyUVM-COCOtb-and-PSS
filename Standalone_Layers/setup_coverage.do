coverage save CNN_top.ucdb -onexit

## Code Coverage Exclusions
coverage exclude -du /CNN_top/pooling_layer_generic -line 102 -code b
coverage exclude -du /CNN_top/pooling_layer_generic -line 102 -code s