	??)b3v@??)b3v@!??)b3v@	??}?t!????}?t!??!??}?t!??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??)b3v@??/EH??A?ԱJi+v@Y?1??l??*	u??ܭA2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?a0,u@!?K?9??X@)?a0,u@1?K?9??X@:Preprocessing2F
Iterator::Model?Y?b+h??!?r?V$-??)????B??1`?w?L???:Preprocessing2P
Iterator::Model::PrefetchN'??rJ??!IV/X?;s?)N'??rJ??1IV/X?;s?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapQ??,u@!?Kݖ?X@))	????q?1????]T?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9??}?t!??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??/EH????/EH??!??/EH??      ??!       "      ??!       *      ??!       2	?ԱJi+v@?ԱJi+v@!?ԱJi+v@:      ??!       B      ??!       J	?1??l???1??l??!?1??l??R      ??!       Z	?1??l???1??l??!?1??l??JCPU_ONLYY??}?t!??b 