	?Z?̄u@?Z?̄u@!?Z?̄u@	?O??
s???O??
s??!?O??
s??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?Z?̄u@M??y ???A*????u@Y?O??e??*	^?I~y?@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?ܵ?|?Q@!ٷj??X@)?ܵ?|?Q@1ٷj??X@:Preprocessing2F
Iterator::Modelk}?Жs??!?B???a??)p?71$'??1I?JE?:Preprocessing2P
Iterator::Model::Prefetch͔?????!????;??)͔?????1????;??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?????Q@!?????X@)?n?h?1????s?p?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?O??
s??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	M??y ???M??y ???!M??y ???      ??!       "      ??!       *      ??!       2	*????u@*????u@!*????u@:      ??!       B      ??!       J	?O??e???O??e??!?O??e??R      ??!       Z	?O??e???O??e??!?O??e??JCPU_ONLYY?O??
s??b 