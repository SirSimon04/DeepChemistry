	<0?? ~@<0?? ~@!<0?? ~@	??R??˽???R??˽?!??R??˽?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$<0?? ~@??,'????AS???}@Y
?5????*	??ʡ?A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??????f@!~E?en?X@)??????f@1~E?en?X@:Preprocessing2F
Iterator::Model??+H3???!D͞CU??)PU??X6??1nì?T???:Preprocessing2P
Iterator::Model::PrefetchC??up???!d??.??)C??up???1d??.??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapˆ5?ōf@!?2a???X@)?????K{?1~??vj+n?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??R??˽?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??,'??????,'????!??,'????      ??!       "      ??!       *      ??!       2	S???}@S???}@!S???}@:      ??!       B      ??!       J	
?5????
?5????!
?5????R      ??!       Z	
?5????
?5????!
?5????JCPU_ONLYY??R??˽?b 