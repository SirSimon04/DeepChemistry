	1AG?J@1AG?J@!1AG?J@	?)???????)??????!?)??????"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$1AG?J@d?g^???Ar3??D@Y???_w???*	??ʡ??A2g
0Iterator::Model::Prefetch::FlatMap[0]::GeneratorW%?}$h@!p,q?6?X@)W%?}$h@1p,q?6?X@:Preprocessing2F
Iterator::Model???WW??!???jz???)F?W?????1m???ݙ??:Preprocessing2P
Iterator::Model::Prefetch?r?9>Z??!??_1sZ}?)?r?9>Z??1??_1sZ}?:Preprocessing2Y
"Iterator::Model::Prefetch::FlatMap8??
*$h@!QY?P?X@)'?_?i?1Pk?$?sZ?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.0% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?)??????#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	d?g^???d?g^???!d?g^???      ??!       "      ??!       *      ??!       2	r3??D@r3??D@!r3??D@:      ??!       B      ??!       J	???_w??????_w???!???_w???R      ??!       Z	???_w??????_w???!???_w???JCPU_ONLYY?)??????b 