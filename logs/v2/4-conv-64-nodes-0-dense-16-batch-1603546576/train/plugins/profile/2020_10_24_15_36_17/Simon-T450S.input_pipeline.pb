	]6:???_@]6:???_@!]6:???_@	??IsT?????IsT???!??IsT???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$]6:???_@?????A??ٮЫ_@Y?F?q????*	??"۳6?@2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorCqǛ??Q@!?rc???X@)CqǛ??Q@1?rc???X@:Preprocessing2F
Iterator::ModelCqǛ???!?I?=??)?˶?ֈ??1? ?	f??:Preprocessing2P
Iterator::Model::PrefetchZ-??DJ??!????[??)Z-??DJ??1????[??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapR???0?Q@!.?޾??X@)?v?$j?1h?Ҟ?r?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??IsT???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??????????!?????      ??!       "      ??!       *      ??!       2	??ٮЫ_@??ٮЫ_@!??ٮЫ_@:      ??!       B      ??!       J	?F?q?????F?q????!?F?q????R      ??!       Z	?F?q?????F?q????!?F?q????JCPU_ONLYY??IsT???b 