	??mnL[n@??mnL[n@!??mnL[n@	=?t??8??=?t??8??!=?t??8??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??mnL[n@??w?}??A??`?HFn@Y????2??*	?Q?C??@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator???Z@!o? ???X@)???Z@1o? ???X@:Preprocessing2F
Iterator::Model]???l??!?7?J?k??)??S?????1??󢹨??:Preprocessing2P
Iterator::Model::Prefetch?4??o???!?%?O????)?4??o???1?%?O????:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?K?e?Z@!9?v??X@) ?K??t?1????ݭs?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9>?t??8??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??w?}????w?}??!??w?}??      ??!       "      ??!       *      ??!       2	??`?HFn@??`?HFn@!??`?HFn@:      ??!       B      ??!       J	????2??????2??!????2??R      ??!       Z	????2??????2??!????2??JCPU_ONLYY>?t??8??b 