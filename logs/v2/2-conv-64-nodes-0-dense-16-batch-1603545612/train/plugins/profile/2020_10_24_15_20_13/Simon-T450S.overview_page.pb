?	?rk?mS]@?rk?mS]@!?rk?mS]@	?_??ȴ??_??ȴ?!?_??ȴ?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?rk?mS]@û\?wb??A*?#??6]@Y?bE?a??*	F??????@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??Y5YQ@!????v?X@)??Y5YQ@1????v?X@:Preprocessing2F
Iterator::Modelu;?ʃ???!??Gش?)-??VФ?1?(_h????:Preprocessing2P
Iterator::Model::Prefetch??ZH??!f	?+?q??)??ZH??1f	?+?q??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap	?L?nYQ@!z9 ???X@)K?8???l?1?K?t?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?_??ȴ?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	û\?wb??û\?wb??!û\?wb??      ??!       "      ??!       *      ??!       2	*?#??6]@*?#??6]@!*?#??6]@:      ??!       B      ??!       J	?bE?a???bE?a??!?bE?a??R      ??!       Z	?bE?a???bE?a??!?bE?a??JCPU_ONLYY?_??ȴ?b Y      Y@q?G7???"?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQ2"CPU: B 