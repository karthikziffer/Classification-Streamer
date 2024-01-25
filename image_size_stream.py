import argparse
import logging
import sys
import numpy as np
import time

from pyflink.common import WatermarkStrategy, Encoder, Types
from pyflink.datastream import StreamExecutionEnvironment, RuntimeExecutionMode
from pyflink.datastream.connectors import (FileSource, StreamFormat, FileSink, OutputFileConfig,
                                           RollingPolicy)

from classifier import image_classifier


def image_dimension(input_path, output_path):
    env = StreamExecutionEnvironment.get_execution_environment()
    env.set_runtime_mode(RuntimeExecutionMode.STREAMING)
    # write all the data to one file
    env.set_parallelism(3)
    # define the source
    
    ds = env.from_source(
        source=FileSource.for_record_stream_format(StreamFormat.text_line_format(),
                                                    input_path)
                            .process_static_file_set().build(),
        watermark_strategy=WatermarkStrategy.for_monotonous_timestamps(),
        source_name="file_source"
    )
    

    # function to split based on line break
    def split(line):
        yield from line.split("\n")

    def classify(row_data):
        # extract pixels
        pixels = list(map(float, row_data.split(',')[-1].split(' ')))
        # pixels to image dimension
        regenerated_image = np.reshape(pixels, (10, 10, 3))
        # model classifier
        image_expended_dim, result, predicted_class, predicted_class_name = image_classifier(regenerated_image)
        # calculate dimension
        h, w, c = regenerated_image.shape
        print(h, w, c)
        return f'height:{h} width:{w} channel:{c} model_result:{result} predicted_class:{predicted_class} predicted_class_name:{predicted_class_name}' 

        
    start = time.time()
    # image classfier
    for _ in range(50):
        print(_)
        ds = ds.flat_map(split) \
            .map(lambda i: classify(i), output_type=Types.STRING())
    stop = time.time()
    print("sinking output")
    print("Processing Time: ", stop - start)
    
    # define the sink
    ds.sink_to(
        sink=FileSink.for_row_format(
            base_path=output_path,
            encoder=Encoder.simple_string_encoder())
        .with_output_file_config(
            OutputFileConfig.builder()
            .build())
        .with_rolling_policy(RollingPolicy.default_rolling_policy())
        .build()
    )
    
    # submit for execution
    env.execute()


if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input',
        dest='input',
        required=False,
        help='Input file to process.')
    parser.add_argument(
        '--output',
        dest='output',
        required=False,
        help='Output file to write results to.')

    argv = sys.argv[1:]
    known_args, _ = parser.parse_known_args(argv)

    image_dimension(known_args.input, known_args.output)
