	?릔?D@?릔?D@!?릔?D@	???@b?E@???@b?E@!???@b?E@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?릔?D@p\?Mt??A?VAtm5@Yz0H??1@*	X9?Ȗ8?@2P
Iterator::Model::Prefetch?c?3??1@!?p?G?X@)?c?3??1@1?p?G?X@:Preprocessing2F
Iterator::Modely?'e?1@!      Y@)H?C??ݠ?17?5p???:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 43.9% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9???@b?E@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	p\?Mt??p\?Mt??!p\?Mt??      ??!       "      ??!       *      ??!       2	?VAtm5@?VAtm5@!?VAtm5@:      ??!       B      ??!       J	z0H??1@z0H??1@!z0H??1@R      ??!       Z	z0H??1@z0H??1@!z0H??1@JCPU_ONLYY???@b?E@b 