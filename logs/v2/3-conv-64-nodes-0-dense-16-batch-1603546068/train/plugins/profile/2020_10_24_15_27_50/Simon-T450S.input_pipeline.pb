	?3w?d_@?3w?d_@!?3w?d_@	@2m???@2m???!@2m???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?3w?d_@??c???A?6ǹMC_@Y??g????*	?Q???@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator ??UyP@!T???X@) ??UyP@1T???X@:Preprocessing2F
Iterator::ModelB?V?9Ν?!?_R Ś??)?el?f??1 K[<??:Preprocessing2P
Iterator::Model::Prefetch|?ԗ????!~sI?y/??)|?ԗ????1~sI?y/??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapxe??yP@!??_?,?X@)A?9w?^j?1ʁ????s?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?2m???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??c?????c???!??c???      ??!       "      ??!       *      ??!       2	?6ǹMC_@?6ǹMC_@!?6ǹMC_@:      ??!       B      ??!       J	??g??????g????!??g????R      ??!       Z	??g??????g????!??g????JCPU_ONLYY?2m???b 