	#?M)??q@#?M)??q@!#?M)??q@	?aq8#???aq8#??!?aq8#??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$#?M)??q@T?~O????A?x@???q@Y???Y???*	?(\??O?@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??=$|?P@!|?2m??X@)??=$|?P@1|?2m??X@:Preprocessing2F
Iterator::ModelU?b?̩?!mg? ?N??)O???|???1]|Ҥ/:??:Preprocessing2P
Iterator::Model::Prefetch[x^*6???!|Rh?~c??)[x^*6???1|Rh?~c??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap1A?°P@!??7J,?X@)n?ݳ?q?1c
?Awz?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?aq8#??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	T?~O????T?~O????!T?~O????      ??!       "      ??!       *      ??!       2	?x@???q@?x@???q@!?x@???q@:      ??!       B      ??!       J	???Y??????Y???!???Y???R      ??!       Z	???Y??????Y???!???Y???JCPU_ONLYY?aq8#??b 