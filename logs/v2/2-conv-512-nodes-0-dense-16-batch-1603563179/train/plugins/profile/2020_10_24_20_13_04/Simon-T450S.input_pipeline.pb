	?p2^?@?p2^?@!?p2^?@	???8Fm????8Fm?!???8Fm?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?p2^?@M??ӀA??Ae?m]?@Y/?N[#???*	?G?z?L?@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?Ü??Y@!$?ƕ?X@)?Ü??Y@1$?ƕ?X@:Preprocessing2F
Iterator::Model}?|?.P??!?C????)???4cє?10_?????:Preprocessing2P
Iterator::Model::Prefetch???H????!?Ž????)???H????1?Ž????:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?W?<??Y@!?w}??X@)x*???Ok?1Ü.?.[j?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???8Fm?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	M??ӀA??M??ӀA??!M??ӀA??      ??!       "      ??!       *      ??!       2	e?m]?@e?m]?@!e?m]?@:      ??!       B      ??!       J	/?N[#???/?N[#???!/?N[#???R      ??!       Z	/?N[#???/?N[#???!/?N[#???JCPU_ONLYY???8Fm?b 