	??????|@??????|@!??????|@	q/????q/????!q/????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??????|@Tt$?????Au:????|@Y??q?j???*	j?t???A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator???]b@!?ib?[?X@)???]b@1?ib?[?X@:Preprocessing2F
Iterator::Model?;?????!???-??)??͋_??1???@????:Preprocessing2P
Iterator::Model::PrefetchZ??լ3??!?ݠcє?)Z??լ3??1?ݠcє?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapA?شRb@!?????X@)?k$	???1??<ݮ,v?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9p/????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Tt$?????Tt$?????!Tt$?????      ??!       "      ??!       *      ??!       2	u:????|@u:????|@!u:????|@:      ??!       B      ??!       J	??q?j?????q?j???!??q?j???R      ??!       Z	??q?j?????q?j???!??q?j???JCPU_ONLYYp/????b 