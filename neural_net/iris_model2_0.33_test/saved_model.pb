??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*	2.5.0-rc12v2.5.0-rc0-36-g0d1805aede08Ã
?
iris_model_2/dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameiris_model_2/dense_3/kernel
?
/iris_model_2/dense_3/kernel/Read/ReadVariableOpReadVariableOpiris_model_2/dense_3/kernel*
_output_shapes

:*
dtype0
?
iris_model_2/dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameiris_model_2/dense_3/bias
?
-iris_model_2/dense_3/bias/Read/ReadVariableOpReadVariableOpiris_model_2/dense_3/bias*
_output_shapes
:*
dtype0
?
iris_model_2/dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameiris_model_2/dense_4/kernel
?
/iris_model_2/dense_4/kernel/Read/ReadVariableOpReadVariableOpiris_model_2/dense_4/kernel*
_output_shapes

:*
dtype0
?
iris_model_2/dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameiris_model_2/dense_4/bias
?
-iris_model_2/dense_4/bias/Read/ReadVariableOpReadVariableOpiris_model_2/dense_4/bias*
_output_shapes
:*
dtype0
?
iris_model_2/dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:*,
shared_nameiris_model_2/dense_5/kernel
?
/iris_model_2/dense_5/kernel/Read/ReadVariableOpReadVariableOpiris_model_2/dense_5/kernel*
_output_shapes

:*
dtype0
?
iris_model_2/dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:**
shared_nameiris_model_2/dense_5/bias
?
-iris_model_2/dense_5/bias/Read/ReadVariableOpReadVariableOpiris_model_2/dense_5/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
hidden0
hidden1
	out_layer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
h

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
 
*
	0

1
2
3
4
5
*
	0

1
2
3
4
5
?
non_trainable_variables
layer_metrics
regularization_losses

layers
metrics
trainable_variables
	variables
layer_regularization_losses
 
ZX
VARIABLE_VALUEiris_model_2/dense_3/kernel)hidden0/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEiris_model_2/dense_3/bias'hidden0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

	0

1

	0

1
?
 non_trainable_variables
!layer_metrics
regularization_losses

"layers
#metrics
trainable_variables
	variables
$layer_regularization_losses
ZX
VARIABLE_VALUEiris_model_2/dense_4/kernel)hidden1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEiris_model_2/dense_4/bias'hidden1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
%non_trainable_variables
&layer_metrics
regularization_losses

'layers
(metrics
trainable_variables
	variables
)layer_regularization_losses
\Z
VARIABLE_VALUEiris_model_2/dense_5/kernel+out_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
XV
VARIABLE_VALUEiris_model_2/dense_5/bias)out_layer/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
*non_trainable_variables
+layer_metrics
regularization_losses

,layers
-metrics
trainable_variables
	variables
.layer_regularization_losses
 
 

0
1
2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1iris_model_2/dense_3/kerneliris_model_2/dense_3/biasiris_model_2/dense_4/kerneliris_model_2/dense_4/biasiris_model_2/dense_5/kerneliris_model_2/dense_5/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_57505
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename/iris_model_2/dense_3/kernel/Read/ReadVariableOp-iris_model_2/dense_3/bias/Read/ReadVariableOp/iris_model_2/dense_4/kernel/Read/ReadVariableOp-iris_model_2/dense_4/bias/Read/ReadVariableOp/iris_model_2/dense_5/kernel/Read/ReadVariableOp-iris_model_2/dense_5/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_57606
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameiris_model_2/dense_3/kerneliris_model_2/dense_3/biasiris_model_2/dense_4/kerneliris_model_2/dense_4/biasiris_model_2/dense_5/kerneliris_model_2/dense_5/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_57634??
?

?
B__inference_dense_3_layer_call_and_return_conditional_losses_57397

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
#__inference_signature_wrapper_57505
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_573822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
'__inference_dense_5_layer_call_fn_57554

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_574312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_5_layer_call_and_return_conditional_losses_57431

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_dense_4_layer_call_fn_57534

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_574142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_57414

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_iris_model_2_layer_call_fn_57456
input_1
unknown:
	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3:
	unknown_4:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_iris_model_2_layer_call_and_return_conditional_losses_574382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
'__inference_dense_3_layer_call_fn_57514

inputs
unknown:
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_573972
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_iris_model_2_layer_call_and_return_conditional_losses_57438
input_1
dense_3_57398:
dense_3_57400:
dense_4_57415:
dense_4_57417:
dense_5_57432:
dense_5_57434:
identity??dense_3/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?dense_5/StatefulPartitionedCall?
dense_3/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_3_57398dense_3_57400*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_573972!
dense_3/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_57415dense_4_57417*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_574142!
dense_4/StatefulPartitionedCall?
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_57432dense_5_57434*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_574312!
dense_5/StatefulPartitionedCall?
IdentityIdentity(dense_5/StatefulPartitionedCall:output:0 ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
B__inference_dense_3_layer_call_and_return_conditional_losses_57525

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
!__inference__traced_restore_57634
file_prefix>
,assignvariableop_iris_model_2_dense_3_kernel::
,assignvariableop_1_iris_model_2_dense_3_bias:@
.assignvariableop_2_iris_model_2_dense_4_kernel::
,assignvariableop_3_iris_model_2_dense_4_bias:@
.assignvariableop_4_iris_model_2_dense_5_kernel::
,assignvariableop_5_iris_model_2_dense_5_bias:

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)hidden0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'hidden0/bias/.ATTRIBUTES/VARIABLE_VALUEB)hidden1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'hidden1/bias/.ATTRIBUTES/VARIABLE_VALUEB+out_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB)out_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp,assignvariableop_iris_model_2_dense_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp,assignvariableop_1_iris_model_2_dense_3_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp.assignvariableop_2_iris_model_2_dense_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp,assignvariableop_3_iris_model_2_dense_4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp.assignvariableop_4_iris_model_2_dense_5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp,assignvariableop_5_iris_model_2_dense_5_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6?

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
__inference__traced_save_57606
file_prefix:
6savev2_iris_model_2_dense_3_kernel_read_readvariableop8
4savev2_iris_model_2_dense_3_bias_read_readvariableop:
6savev2_iris_model_2_dense_4_kernel_read_readvariableop8
4savev2_iris_model_2_dense_4_bias_read_readvariableop:
6savev2_iris_model_2_dense_5_kernel_read_readvariableop8
4savev2_iris_model_2_dense_5_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B)hidden0/kernel/.ATTRIBUTES/VARIABLE_VALUEB'hidden0/bias/.ATTRIBUTES/VARIABLE_VALUEB)hidden1/kernel/.ATTRIBUTES/VARIABLE_VALUEB'hidden1/bias/.ATTRIBUTES/VARIABLE_VALUEB+out_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEB)out_layer/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:06savev2_iris_model_2_dense_3_kernel_read_readvariableop4savev2_iris_model_2_dense_3_bias_read_readvariableop6savev2_iris_model_2_dense_4_kernel_read_readvariableop4savev2_iris_model_2_dense_4_bias_read_readvariableop6savev2_iris_model_2_dense_5_kernel_read_readvariableop4savev2_iris_model_2_dense_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*G
_input_shapes6
4: ::::::: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::$ 

_output_shapes

:: 

_output_shapes
::

_output_shapes
: 
?

?
B__inference_dense_4_layer_call_and_return_conditional_losses_57545

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?&
?
 __inference__wrapped_model_57382
input_1E
3iris_model_2_dense_3_matmul_readvariableop_resource:B
4iris_model_2_dense_3_biasadd_readvariableop_resource:E
3iris_model_2_dense_4_matmul_readvariableop_resource:B
4iris_model_2_dense_4_biasadd_readvariableop_resource:E
3iris_model_2_dense_5_matmul_readvariableop_resource:B
4iris_model_2_dense_5_biasadd_readvariableop_resource:
identity??+iris_model_2/dense_3/BiasAdd/ReadVariableOp?*iris_model_2/dense_3/MatMul/ReadVariableOp?+iris_model_2/dense_4/BiasAdd/ReadVariableOp?*iris_model_2/dense_4/MatMul/ReadVariableOp?+iris_model_2/dense_5/BiasAdd/ReadVariableOp?*iris_model_2/dense_5/MatMul/ReadVariableOp?
*iris_model_2/dense_3/MatMul/ReadVariableOpReadVariableOp3iris_model_2_dense_3_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*iris_model_2/dense_3/MatMul/ReadVariableOp?
iris_model_2/dense_3/MatMulMatMulinput_12iris_model_2/dense_3/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
iris_model_2/dense_3/MatMul?
+iris_model_2/dense_3/BiasAdd/ReadVariableOpReadVariableOp4iris_model_2_dense_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+iris_model_2/dense_3/BiasAdd/ReadVariableOp?
iris_model_2/dense_3/BiasAddBiasAdd%iris_model_2/dense_3/MatMul:product:03iris_model_2/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
iris_model_2/dense_3/BiasAdd?
iris_model_2/dense_3/ReluRelu%iris_model_2/dense_3/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
iris_model_2/dense_3/Relu?
*iris_model_2/dense_4/MatMul/ReadVariableOpReadVariableOp3iris_model_2_dense_4_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*iris_model_2/dense_4/MatMul/ReadVariableOp?
iris_model_2/dense_4/MatMulMatMul'iris_model_2/dense_3/Relu:activations:02iris_model_2/dense_4/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
iris_model_2/dense_4/MatMul?
+iris_model_2/dense_4/BiasAdd/ReadVariableOpReadVariableOp4iris_model_2_dense_4_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+iris_model_2/dense_4/BiasAdd/ReadVariableOp?
iris_model_2/dense_4/BiasAddBiasAdd%iris_model_2/dense_4/MatMul:product:03iris_model_2/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
iris_model_2/dense_4/BiasAdd?
iris_model_2/dense_4/ReluRelu%iris_model_2/dense_4/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
iris_model_2/dense_4/Relu?
*iris_model_2/dense_5/MatMul/ReadVariableOpReadVariableOp3iris_model_2_dense_5_matmul_readvariableop_resource*
_output_shapes

:*
dtype02,
*iris_model_2/dense_5/MatMul/ReadVariableOp?
iris_model_2/dense_5/MatMulMatMul'iris_model_2/dense_4/Relu:activations:02iris_model_2/dense_5/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
iris_model_2/dense_5/MatMul?
+iris_model_2/dense_5/BiasAdd/ReadVariableOpReadVariableOp4iris_model_2_dense_5_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+iris_model_2/dense_5/BiasAdd/ReadVariableOp?
iris_model_2/dense_5/BiasAddBiasAdd%iris_model_2/dense_5/MatMul:product:03iris_model_2/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
iris_model_2/dense_5/BiasAdd?
iris_model_2/dense_5/SoftmaxSoftmax%iris_model_2/dense_5/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
iris_model_2/dense_5/Softmax?
IdentityIdentity&iris_model_2/dense_5/Softmax:softmax:0,^iris_model_2/dense_3/BiasAdd/ReadVariableOp+^iris_model_2/dense_3/MatMul/ReadVariableOp,^iris_model_2/dense_4/BiasAdd/ReadVariableOp+^iris_model_2/dense_4/MatMul/ReadVariableOp,^iris_model_2/dense_5/BiasAdd/ReadVariableOp+^iris_model_2/dense_5/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : : : : : 2Z
+iris_model_2/dense_3/BiasAdd/ReadVariableOp+iris_model_2/dense_3/BiasAdd/ReadVariableOp2X
*iris_model_2/dense_3/MatMul/ReadVariableOp*iris_model_2/dense_3/MatMul/ReadVariableOp2Z
+iris_model_2/dense_4/BiasAdd/ReadVariableOp+iris_model_2/dense_4/BiasAdd/ReadVariableOp2X
*iris_model_2/dense_4/MatMul/ReadVariableOp*iris_model_2/dense_4/MatMul/ReadVariableOp2Z
+iris_model_2/dense_5/BiasAdd/ReadVariableOp+iris_model_2/dense_5/BiasAdd/ReadVariableOp2X
*iris_model_2/dense_5/MatMul/ReadVariableOp*iris_model_2/dense_5/MatMul/ReadVariableOp:P L
'
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
B__inference_dense_5_layer_call_and_return_conditional_losses_57565

inputs0
matmul_readvariableop_resource:-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdda
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????2	
Softmax?
IdentityIdentitySoftmax:softmax:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
;
input_10
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????tensorflow/serving/predict:?U
?
hidden0
hidden1
	out_layer
regularization_losses
trainable_variables
	variables
	keras_api

signatures
/_default_save_signature
0__call__
*1&call_and_return_all_conditional_losses"?
_tf_keras_model?{"name": "iris_model_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "IrisModel", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [100, 4]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "IrisModel"}}
?	

	kernel

bias
regularization_losses
trainable_variables
	variables
	keras_api
2__call__
*3&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [100, 4]}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
4__call__
*5&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "units": 8, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [100, 8]}}
?

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
6__call__
*7&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 3, "activation": "softmax", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [100, 8]}}
 "
trackable_list_wrapper
J
	0

1
2
3
4
5"
trackable_list_wrapper
J
	0

1
2
3
4
5"
trackable_list_wrapper
?
non_trainable_variables
layer_metrics
regularization_losses

layers
metrics
trainable_variables
	variables
layer_regularization_losses
0__call__
/_default_save_signature
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
,
8serving_default"
signature_map
-:+2iris_model_2/dense_3/kernel
':%2iris_model_2/dense_3/bias
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
?
 non_trainable_variables
!layer_metrics
regularization_losses

"layers
#metrics
trainable_variables
	variables
$layer_regularization_losses
2__call__
*3&call_and_return_all_conditional_losses
&3"call_and_return_conditional_losses"
_generic_user_object
-:+2iris_model_2/dense_4/kernel
':%2iris_model_2/dense_4/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
%non_trainable_variables
&layer_metrics
regularization_losses

'layers
(metrics
trainable_variables
	variables
)layer_regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
-:+2iris_model_2/dense_5/kernel
':%2iris_model_2/dense_5/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
*non_trainable_variables
+layer_metrics
regularization_losses

,layers
-metrics
trainable_variables
	variables
.layer_regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
 __inference__wrapped_model_57382?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
,__inference_iris_model_2_layer_call_fn_57456?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
G__inference_iris_model_2_layer_call_and_return_conditional_losses_57438?
???
FullArgSpec
args?
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *&?#
!?
input_1?????????
?2?
'__inference_dense_3_layer_call_fn_57514?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_3_layer_call_and_return_conditional_losses_57525?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_4_layer_call_fn_57534?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_4_layer_call_and_return_conditional_losses_57545?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_dense_5_layer_call_fn_57554?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
B__inference_dense_5_layer_call_and_return_conditional_losses_57565?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
#__inference_signature_wrapper_57505input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
 __inference__wrapped_model_57382o	
0?-
&?#
!?
input_1?????????
? "3?0
.
output_1"?
output_1??????????
B__inference_dense_3_layer_call_and_return_conditional_losses_57525\	
/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_3_layer_call_fn_57514O	
/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dense_4_layer_call_and_return_conditional_losses_57545\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_4_layer_call_fn_57534O/?,
%?"
 ?
inputs?????????
? "???????????
B__inference_dense_5_layer_call_and_return_conditional_losses_57565\/?,
%?"
 ?
inputs?????????
? "%?"
?
0?????????
? z
'__inference_dense_5_layer_call_fn_57554O/?,
%?"
 ?
inputs?????????
? "???????????
G__inference_iris_model_2_layer_call_and_return_conditional_losses_57438a	
0?-
&?#
!?
input_1?????????
? "%?"
?
0?????????
? ?
,__inference_iris_model_2_layer_call_fn_57456T	
0?-
&?#
!?
input_1?????????
? "???????????
#__inference_signature_wrapper_57505z	
;?8
? 
1?.
,
input_1!?
input_1?????????"3?0
.
output_1"?
output_1?????????