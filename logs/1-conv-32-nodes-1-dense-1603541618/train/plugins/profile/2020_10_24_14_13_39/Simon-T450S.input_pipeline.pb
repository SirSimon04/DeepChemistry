	 ?M??cR@ ?M??cR@! ?M??cR@	n??? @n??? @!n??? @"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ ?M??cR@??l#???A??u6?Q@Y?[???@*	    `L?@2P
Iterator::Model::Prefetch'???@!?bW??X@)'???@1?bW??X@:Preprocessing2F
Iterator::Modelm?????@!      Y@)?i? ?Ӛ?1??N?????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 6.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*no9n??? @>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??l#?????l#???!??l#???      ??!       "      ??!       *      ??!       2	??u6?Q@??u6?Q@!??u6?Q@:      ??!       B      ??!       J	?[???@?[???@!?[???@R      ??!       Z	?[???@?[???@!?[???@JCPU_ONLYYn??? @b 