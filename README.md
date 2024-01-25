# Real-time image classifier using Apache Flink

1. Project Objective:
    - Develop an Apache Flink real-time streaming service for a Machine Learning Application.
    - Conduct image classification on the incoming pixel stream from the source.
    - Record the prediction results and send them to the output destination (sink).

2. Secondary Goal:
    - Compare the performance of Apache Flink with a Python script for the image classification task.

3. Experimental Approach:
    - Conduct multiple trials (5 in total) to ensure consistency in results.
    - Calculate the average result from the trials.

4. Conclusion:
    - Findings indicate that the execution time using Apache Flink is faster.
    - Highlight additional advantages, such as parallelization, achieved through Apache Flink.


## Results

![](./assets/Average_execution_time.png = 250x250)

![](.\assets\Execution time comparision.png)
