# PySyun Time-Line Harmonics

This library provides tools for analyzing time-based transaction data using Fourier analysis to identify periodic patterns and intensity cycles. It contains two main processor classes that offer different approaches to Fourier analysis.

## Overview

The library enables you to:
- Analyze transaction intensity over time
- Identify dominant periodic patterns in transaction data
- Perform sliding window analysis to detect localized patterns
- Measure the significance and consistency of detected periods

## Installation

```bash
pip install git+https://github.com/pysyun/pysyun_timeline_harmonics.git
```

## Classes

The library provides two main classes:

### 1. FourierIntensityProcessor

This class performs basic Fourier analysis on the entire dataset to identify dominant periods.

#### Key Features:
- Automatically determines sampling rate from transaction data
- Bins transactions into intensity time series
- Identifies top periodic patterns based on power spectrum
- Simple and efficient for datasets with consistent patterns

### 2. SlidingWindowFourierProcessor

This advanced processor applies Fourier analysis with multiple sliding windows, allowing it to detect both global and localized patterns.

#### Key Features:
- Adaptive window size selection
- Multiple overlapping windows for better time resolution
- Filters periods based on consistency across windows
- Detects whether patterns are localized in time or consistent throughout
- Calculates significance metrics for each detected period

## Usage Examples

### Basic Fourier Analysis

```python
import numpy as np
from fourier_transaction_analysis import FourierIntensityProcessor

# Example transaction data (list of dictionaries with 'time' field)
transactions = [
    {'time': 1000.0, 'value': 10},
    {'time': 1005.2, 'value': 15},
    # ... more transactions
]

# Initialize the processor
processor = FourierIntensityProcessor(num_peaks=5)

# Process the data
results = processor.process(transactions)

# Access the results
for pattern in results:
    print(f"Period: {pattern['period']} seconds")
    print(f"Frequency: {pattern['frequency']} Hz")
    print(f"Power: {pattern['power_spectrum']}")
```

### Sliding Window Analysis

```python
from fourier_transaction_analysis import SlidingWindowFourierProcessor

# Initialize the sliding window processor with custom parameters
processor = SlidingWindowFourierProcessor(
    min_window_size=100,
    max_window_size=1000, 
    window_overlap=0.5,
    num_peaks=5,
    min_peak_height=0.3,
    consistency_threshold=0.7
)

# Process the transaction data
results = processor.process(transactions)

# Access the detected periods
for period_info in results['periods']:
    print(f"Period: {period_info['period']} seconds")
    print(f"Significance: {period_info['significance']}")
    print(f"Occurrence frequency: {period_info['occurrence_frequency']}")
    print(f"Is localized: {period_info['is_localized']}")
    if period_info['is_localized']:
        print(f"Time range: {period_info['time_range']}")
```

## Advanced Configuration

### FourierIntensityProcessor Parameters

- `sampling_rate` (optional): Custom sampling rate. If not provided, it will be calculated from the median time interval between transactions.
- `num_peaks`: Number of dominant peaks to identify in the spectrum.

### SlidingWindowFourierProcessor Parameters

- `min_window_size`: Minimum size of the sliding window (in samples).
- `max_window_size`: Maximum size of the sliding window (in samples).
- `window_overlap`: Fraction of overlap between consecutive windows (0.0-1.0).
- `num_peaks`: Maximum number of peaks to detect in each window.
- `min_peak_height`: Minimum relative height of peaks (as fraction of maximum power).
- `consistency_threshold`: Minimum fraction of windows where a period must appear to be considered significant.
- `sampling_rate` (optional): Custom sampling rate. If not provided, it will be calculated automatically.

## Data Format

The input data for both processors should be a list of dictionaries, where each dictionary represents a transaction and contains at least a `time` field with the timestamp of the transaction:

```python
transactions = [
    {'time': 1000.0, 'other_field': 'value1'},
    {'time': 1002.5, 'other_field': 'value2'},
    # ... more transactions
]
```

## Output Interpretation

### FourierIntensityProcessor Results

Returns a list of dictionaries, each containing:
- `period`: Duration of the cycle in the same time units as the input data
- `frequency`: The frequency of the cycle (1/period)
- `power_spectrum`: Power value representing the strength of the period
- `time`: Array of time points used in the analysis
- `value`: Array of transaction counts at each time point

### SlidingWindowFourierProcessor Results

Returns a dictionary with a `periods` key containing a list of dictionaries, each with:
- `period`: Duration of the cycle
- `occurrence_frequency`: Fraction of windows where this period was detected
- `average_power`: Average power across all detections
- `max_power`: Maximum power observed for this period
- `significance`: Combined metric of frequency and power
- `time_range`: (start, end) range where the period was detected
- `duration`: Time span where the period was active
- `is_localized`: Boolean indicating if the period is localized in time
- `time_points`: List of time points where the period was detected
- `power_values`: Power values at each detection point
- `optimal_window_size`: Best window size for detecting this period

## Notes

- The library uses numpy and scipy for efficient computation.
- For large datasets, consider adjusting window sizes to balance between detection accuracy and computation time.
- Periods that are longer than half the total data duration may not be reliable.

## License

MIT