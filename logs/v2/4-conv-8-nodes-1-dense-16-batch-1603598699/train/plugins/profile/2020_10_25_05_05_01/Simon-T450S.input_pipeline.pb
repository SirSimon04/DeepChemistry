	??l?%?K@??l?%?K@!??l?%?K@	unJ ?4D@unJ ?4D@!unJ ?4D@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??l?%?K@??s?//??AqǛ???@Y???A?6@*	?O??v??@2P
Iterator::Model::Prefetch?T?:v6@!???:M?X@)?T?:v6@1???:M?X@:Preprocessing2F
Iterator::Model?zܷZ{6@!      Y@)??t!V??1P]	|˶?:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
host?Your program is HIGHLY input-bound because 40.4% of the total step time sampled is waiting for input. Therefore, you should first focus on reducing the input time.no*no9tnJ ?4D@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	??s?//????s?//??!??s?//??      ??!       "      ??!       *      ??!       2	qǛ???@qǛ???@!qǛ???@:      ??!       B      ??!       J	???A?6@???A?6@!???A?6@R      ??!       Z	???A?6@???A?6@!???A?6@JCPU_ONLYYtnJ ?4D@b 