	h?????@h?????@!h?????@	f1*?O???f1*?O???!f1*?O???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$h?????@????????A??q?@Y@?0`?U??*	L7?AS?A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?҈??!b@!?@???X@)?҈??!b@1?@???X@:Preprocessing2F
Iterator::Model??	ܺ???!&S5????)??o'???1g??????:Preprocessing2P
Iterator::Model::Prefetch?ΤMե?!??1f!??)?ΤMե?1??1f!??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?5=(("b@!???W?X@)???1ZGu?1???5?m?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9e1*?O???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????????????????!????????      ??!       "      ??!       *      ??!       2	??q?@??q?@!??q?@:      ??!       B      ??!       J	@?0`?U??@?0`?U??!@?0`?U??R      ??!       Z	@?0`?U??@?0`?U??!@?0`?U??JCPU_ONLYYe1*?O???b 