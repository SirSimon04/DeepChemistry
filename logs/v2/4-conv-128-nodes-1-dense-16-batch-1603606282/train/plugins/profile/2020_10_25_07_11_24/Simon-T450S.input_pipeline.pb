	?C??~,o@?C??~,o@!?C??~,o@	7????ˣ?7????ˣ?!7????ˣ?"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?C??~,o@??	L????A??I`so@Y#/kb????*	P??n???@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator??v??4Q@!/y???X@)??v??4Q@1/y???X@:Preprocessing2F
Iterator::Model??@??_??!??)_+ķ?)??U????1D??3???:Preprocessing2P
Iterator::Model::Prefetch?
E????!&e\?{??)?
E????1&e\?{??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapܷZ'.5Q@!?5(??X@)R?r?n?1_^/?=v?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no97????ˣ?#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??	L??????	L????!??	L????      ??!       "      ??!       *      ??!       2	??I`so@??I`so@!??I`so@:      ??!       B      ??!       J	#/kb????#/kb????!#/kb????R      ??!       Z	#/kb????#/kb????!#/kb????JCPU_ONLYY7????ˣ?b 