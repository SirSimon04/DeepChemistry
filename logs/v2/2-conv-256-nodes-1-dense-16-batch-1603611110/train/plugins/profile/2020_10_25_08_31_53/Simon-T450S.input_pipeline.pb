	??ӹ"J?@??ӹ"J?@!??ӹ"J?@		???r???	???r???!	???r???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??ӹ"J?@Iط????AE|6C?@Y?i3NCT??*	?K7???@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator?-z?X@!??l?	?X@)?-z?X@1??l?	?X@:Preprocessing2F
Iterator::ModelG!ɬ????!Cu;/???)rm??o??1?/zE{???:Preprocessing2P
Iterator::Model::PrefetchK=By??!?W?nf??)K=By??1?W?nf??:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMapg,??N?X@!?????X@)Qj/?혊?1??y~????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9	???r???#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	Iط????Iط????!Iط????      ??!       "      ??!       *      ??!       2	E|6C?@E|6C?@!E|6C?@:      ??!       B      ??!       J	?i3NCT???i3NCT??!?i3NCT??R      ??!       Z	?i3NCT???i3NCT??!?i3NCT??JCPU_ONLYY	???r???b 