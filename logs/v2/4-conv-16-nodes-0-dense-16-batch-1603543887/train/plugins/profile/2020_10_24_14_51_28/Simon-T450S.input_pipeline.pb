	????BJ@????BJ@!????BJ@	Gq1?o;3@Gq1?o;3@!Gq1?o;3@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$????BJ@?h?'????A%??W??D@Yq?GR?3$@*	)\??U??@2P
Iterator::Model::Prefetch^f?(?$@!Q?0ؔ?X@)^f?(?$@1Q?0ؔ?X@:Preprocessing2F
Iterator::Model?̒ 5%$@!      Y@)??f??I??1y?n?'k??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 19.2% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9Gq1?o;3@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	?h?'?????h?'????!?h?'????      ??!       "      ??!       *      ??!       2	%??W??D@%??W??D@!%??W??D@:      ??!       B      ??!       J	q?GR?3$@q?GR?3$@!q?GR?3$@R      ??!       Z	q?GR?3$@q?GR?3$@!q?GR?3$@JCPU_ONLYYGq1?o;3@b 