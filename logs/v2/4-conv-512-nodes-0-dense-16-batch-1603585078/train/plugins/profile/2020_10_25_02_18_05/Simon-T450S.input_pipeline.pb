	?jIi?@?jIi?@!?jIi?@	???5?i????5?i?!???5?i?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?jIi?@????????A?'v?f?@Y?{ds?<??*	??????@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?rJ@LN\@!%?R??X@)?rJ@LN\@1%?R??X@:Preprocessing2F
Iterator::Model?"nN%??!?i8M????)4Lm?????1?%˼ "??:Preprocessing2P
Iterator::Model::Prefetch???2???!٭??˔?)???2???1٭??˔?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap?E_A?N\@!?X?-?X@)?v?4E?s?1!a???6q?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9???5?i?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	????????????????!????????      ??!       "      ??!       *      ??!       2	?'v?f?@?'v?f?@!?'v?f?@:      ??!       B      ??!       J	?{ds?<???{ds?<??!?{ds?<??R      ??!       Z	?{ds?<???{ds?<??!?{ds?<??JCPU_ONLYY???5?i?b 