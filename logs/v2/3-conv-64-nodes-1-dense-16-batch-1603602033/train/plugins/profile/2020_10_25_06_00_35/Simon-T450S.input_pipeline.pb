	?j?:`@?j?:`@!?j?:`@	/q5??*??/q5??*??!/q5??*??"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?j?:`@3#????AHp#e,`@Y??????*	w????@2g
0Iterator::Model::Prefetch::FlatMap[0]::Generator???kz~Q@! Lsr?X@)???kz~Q@1 Lsr?X@:Preprocessing2F
Iterator::Model??:???!??ڄ?V??)w?$???1ySs????:Preprocessing2P
Iterator::Model::PrefetchOyt#,*??!?F?,7???)Oyt#,*??1?F?,7???:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap(}!??~Q@!Z??\j?X@)2??p?1??Vߚ?w?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9/q5??*??#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	3#????3#????!3#????      ??!       "      ??!       *      ??!       2	Hp#e,`@Hp#e,`@!Hp#e,`@:      ??!       B      ??!       J	????????????!??????R      ??!       Z	????????????!??????JCPU_ONLYY/q5??*??b 