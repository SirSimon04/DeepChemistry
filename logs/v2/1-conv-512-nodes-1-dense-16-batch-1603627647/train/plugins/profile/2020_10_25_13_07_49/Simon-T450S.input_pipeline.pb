	|,}???@|,}???@!|,}???@	? ??G???? ??G???!? ??G???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$|,}???@??Z`?)@A?O?????@Y??1??%@*	?x?&7?A2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??j?o@!և0P?X@)??j?o@1և0P?X@:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap???o@!??F(??X@)??**??1ƪ????:Preprocessing2P
Iterator::Model::Prefetch?'?H0???!??裷??)?'?H0???1??裷??:Preprocessing2F
Iterator::Model'??2???!;l??/??)??m5???1jAS????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9? ??G???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??Z`?)@??Z`?)@!??Z`?)@      ??!       "      ??!       *      ??!       2	?O?????@?O?????@!?O?????@:      ??!       B      ??!       J	??1??%@??1??%@!??1??%@R      ??!       Z	??1??%@??1??%@!??1??%@JCPU_ONLYY? ??G???b 