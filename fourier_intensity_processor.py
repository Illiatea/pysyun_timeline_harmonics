import numpy as np
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks, windows
from collections import defaultdict
from astropy.timeseries import LombScargle

class FourierIntensityProcessor:
    def __init__(self, sampling_rate=None, num_peaks=5):
        """
        Initialize the processor for analyzing trade intensity using Fourier analysis.

        Args:
            sampling_rate (float, optional): The sampling rate. If not provided, the median interval between transactions will be used.
            num_peaks (int): Number of peaks in the spectrum to search for dominant periods.
        """
        self.sampling_rate = sampling_rate
        self.num_peaks = num_peaks

    def _prepare_data(self, data):
        """
        Prepare time data from transactions.

        Args:
            data (list): List of transactions, each with a 'time' field.

        Returns:
            tuple: (times, sampling_rate, dt) – sorted time stamps, selected sampling rate, and time step (dt).
        """
        times = np.array([tx['time'] for tx in data])
        times = np.sort(times)

        if self.sampling_rate is None:
            intervals = np.diff(times)
            dt = np.median(intervals)
            sampling_rate = 1.0 / dt
        else:
            sampling_rate = self.sampling_rate
            dt = 1.0 / sampling_rate

        return times, sampling_rate, dt

    def _bin_data(self, times, dt):
        """
        Convert a list of time stamps into a time series of intensity (number of transactions per interval).

        Args:
            times (np.array): Sorted time stamps.
            dt (float): Time step (sampling interval).

        Returns:
            tuple: (bin_centers, counts) – centers of the time bins and transaction counts in each bin.
        """
        t_start = times[0]
        t_end = times[-1]
        bin_edges = np.arange(t_start, t_end + dt, dt)
        counts, _ = np.histogram(times, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        return bin_centers, counts

    def _find_periods(self, freqs, power_spectrum):
        """
        Find the dominant periods in the power spectrum.

        Args:
            freqs (np.array): Array of frequencies.
            power_spectrum (np.array): Power spectrum values.

        Returns:
            list: List of tuples (period, frequency, power) for each detected peak, sorted by descending power.
        """
        peaks, properties = find_peaks(power_spectrum, height=0)
        if len(peaks) == 0:
            return []

        peak_heights = properties['peak_heights']
        sorted_indices = np.argsort(peak_heights)[::-1]
        top_peaks = peaks[sorted_indices[:self.num_peaks]]

        peak_info = []
        for peak in top_peaks:
            freq = freqs[peak]
            period = 1.0 / freq
            power = power_spectrum[peak]
            peak_info.append((period, freq, power))

        # Sort by descending power
        peak_info = sorted(peak_info, key=lambda x: x[2], reverse=True)
        return peak_info

    def process(self, data):
        """
        Perform Fourier analysis on trade intensity data.

        Args:
            data (list): List of transactions, where each transaction contains a 'time' field.

        Returns:
            list: An array of dictionaries in the following format:
            [
                {
                    'period': period,
                    'power_spectrum': power,
                    'frequency': frequency,
                    'time': time_series,
                    'value': counts
                },
                ...
            ]
        """
        times, sampling_rate, dt = self._prepare_data(data)
        time_series, counts = self._bin_data(times, dt)
        n = len(counts)

        fft_values = fft(counts)
        freqs = fftfreq(n, dt)

        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        power_spectrum = np.abs(fft_values[pos_mask])**2

        peak_info = self._find_periods(freqs, power_spectrum)

        result = []
        for period, frequency, power in peak_info:
            result.append({
                'period': period,
                'power_spectrum': power,
                'frequency': frequency,
                'time': time_series,
                'value': counts
            })

        return result


class SlidingWindowFourierProcessor:
    def __init__(self, min_window_size=100, max_window_size=1000,
                 window_overlap=0.5, num_peaks=5, min_peak_height=0.3,
                 consistency_threshold=0.7, sampling_rate=None):
        """
        Initialize the processor for analyzing transaction intensity using
        sliding window Fourier analysis.

        Parameters:
            min_window_size (int): Minimum size of sliding window (in samples)
            max_window_size (int): Maximum size of sliding window (in samples)
            window_overlap (float): Fraction of overlap between consecutive windows (0.0-1.0)
            num_peaks (int): Maximum number of peaks to detect in each window
            min_peak_height (float): Minimum relative height of peaks (as a fraction of maximum power)
            consistency_threshold (float): Minimum fraction of windows where a period must appear
            sampling_rate (float, optional): Sampling rate. If not provided, the median interval
                                           between transactions will be used.
        """
        self.min_window_size = min_window_size
        self.max_window_size = max_window_size
        self.window_overlap = window_overlap
        self.num_peaks = num_peaks
        self.min_peak_height = min_peak_height
        self.consistency_threshold = consistency_threshold
        self.sampling_rate = sampling_rate

    def _prepare_data(self, data):
        """
        Prepare time data from transactions.

        Parameters:
            data (list): List of transactions, each with a 'time' field.

        Returns:
            tuple: (time_series, counts, dt) – time series, counts, and time step
        """
        times = np.array([tx['time'] for tx in data])
        times = np.sort(times)

        if self.sampling_rate is None:
            intervals = np.diff(times)
            dt = np.median(intervals)
            sampling_rate = 1.0 / dt
        else:
            sampling_rate = self.sampling_rate
            dt = 1.0 / sampling_rate

        t_start = times[0]
        t_end = times[-1]
        bin_edges = np.arange(t_start, t_end + dt, dt)
        counts, _ = np.histogram(times, bins=bin_edges)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

        return bin_centers, counts, dt

    def _apply_window(self, counts, window_size, start_idx):
        """
        Apply a window function to a segment of data.

        Parameters:
            counts (np.array): Full array of samples
            window_size (int): Window size to apply
            start_idx (int): Starting index for the window

        Returns:
            tuple: (windowed_data, window_center) - windowed data and center index
        """
        end_idx = min(start_idx + window_size, len(counts))
        actual_size = end_idx - start_idx

        if actual_size < window_size / 2:  # Skip if window is too small
            return None, None

        # Apply a Hann window to reduce spectral leakage
        windowed_data = counts[start_idx:end_idx] * windows.hann(actual_size)
        window_center = start_idx + actual_size // 2

        return windowed_data, window_center

    def _find_periods_in_window(self, windowed_data, dt, window_center=None):
        """
        Find dominant periods in a windowed data segment.

        Parameters:
            windowed_data (np.array): Windowed data segment
            dt (float): Time step
            window_center (int, optional): Center index of window for time localization

        Returns:
            list: List of tuples (period, power, window_center) for detected peaks
        """
        if windowed_data is None or len(windowed_data) < 10:
            return []

        n = len(windowed_data)
        fft_values = fft(windowed_data)
        freqs = fftfreq(n, dt)

        # Keep only positive frequencies
        pos_mask = freqs > 0
        freqs = freqs[pos_mask]
        power_spectrum = np.abs(fft_values[pos_mask]) ** 2

        # Normalize power spectrum
        if np.max(power_spectrum) > 0:
            power_spectrum = power_spectrum / np.max(power_spectrum)

        # Find peaks with minimum height
        peaks, properties = find_peaks(power_spectrum, height=self.min_peak_height)

        if len(peaks) == 0:
            return []

        # Sort peaks by power and select the best ones
        peak_heights = properties['peak_heights']
        sorted_indices = np.argsort(peak_heights)[::-1]
        top_peaks = peaks[sorted_indices[:self.num_peaks]]

        peak_info = []
        for peak in top_peaks:
            freq = freqs[peak]
            # Skip very low frequencies (long periods that may be unreliable)
            if freq < 1.0 / (n * dt * 0.5):
                continue
            period = 1.0 / freq
            power = power_spectrum[peak]
            peak_info.append((period, power, window_center))

        return peak_info

    def _adaptive_window_analysis(self, counts, time_series, dt):
        """
        Apply Fourier analysis with multiple window sizes and track where periods appear.

        Parameters:
            counts (np.array): Array of transaction counts
            time_series (np.array): Time points corresponding to the samples
            dt (float): Time step

        Returns:
            tuple: (all_periods, period_occurrences, period_locations)
                - all_periods: List of all detected (period, power, window_center)
                - period_occurrences: Dictionary mapping periods to occurrence counts
                - period_locations: Dictionary mapping periods to lists of time positions
        """
        all_periods = []
        period_occurrences = defaultdict(int)
        period_locations = defaultdict(list)
        total_windows = 0

        # Try different window sizes
        window_sizes = np.linspace(self.min_window_size, self.max_window_size, 5).astype(int)

        for window_size in window_sizes:
            step_size = max(1, int(window_size * (1 - self.window_overlap)))

            # Slide window over the data
            for start_idx in range(0, len(counts) - window_size + 1, step_size):
                windowed_data, window_center = self._apply_window(counts, window_size, start_idx)
                if windowed_data is None:
                    continue

                periods = self._find_periods_in_window(windowed_data, dt, window_center)

                total_windows += 1

                # Track all periods and where they appear
                for period, power, center_idx in periods:
                    # Group similar periods (within 10%)
                    grouped = False
                    for existing_period in list(period_occurrences.keys()):
                        if abs(existing_period - period) / existing_period < 0.1:
                            period_occurrences[existing_period] += 1
                            if center_idx is not None:
                                period_locations[existing_period].append(
                                    (time_series[center_idx], power, window_size)
                                )
                            grouped = True
                            break

                    if not grouped:
                        period_occurrences[period] = 1
                        if center_idx is not None:
                            period_locations[period] = [
                                (time_series[center_idx], power, window_size)
                            ]

                    all_periods.append((period, power, center_idx))

        # Calculate occurrence frequencies
        for period in list(period_occurrences.keys()):
            period_occurrences[period] /= max(1, total_windows)

        return all_periods, period_occurrences, period_locations

    def _filter_and_analyze_periods(self, period_occurrences, period_locations):
        """
        Filter periods by consistency and analyze their temporal distribution.

        Parameters:
            period_occurrences (dict): Mapping of periods to their frequency
            period_locations (dict): Mapping of periods to places where they were detected

        Returns:
            list: Filtered list of dictionaries with period information
        """
        filtered_periods = []

        for period, occurrence_freq in period_occurrences.items():
            if occurrence_freq >= self.consistency_threshold:
                locations = period_locations.get(period, [])

                if locations:
                    # Extract times and powers
                    times, powers, window_sizes = zip(*locations)

                    # Calculate statistics
                    avg_power = np.mean(powers)
                    max_power = np.max(powers)
                    time_range = (min(times), max(times))
                    duration = time_range[1] - time_range[0]

                    # Check if period is localized or distributed
                    time_std = np.std(times) if len(times) > 1 else 0
                    is_localized = time_std < (time_range[1] - time_range[0]) / 4

                    # Optimal window size for this period
                    optimal_window_size = np.mean(window_sizes)

                    filtered_periods.append({
                        'period': period,
                        'occurrence_frequency': occurrence_freq,
                        'average_power': avg_power,
                        'max_power': max_power,
                        'significance': occurrence_freq * avg_power,
                        'time_range': time_range,
                        'duration': duration,
                        'is_localized': is_localized,
                        'time_points': times,
                        'power_values': powers,
                        'optimal_window_size': optimal_window_size
                    })

        # Sort by significance
        filtered_periods.sort(key=lambda x: x['significance'], reverse=True)

        return filtered_periods

    def process(self, data):
        """
        Process transaction data using sliding window Fourier analysis to find periods.

        Parameters:
            data (list): List of transactions, each with a 'time' field

        Returns:
            dict: Results containing:
                - 'periods': List of detected periods with their properties
        """
        time_series, counts, dt = self._prepare_data(data)

        # Apply sliding windows of various sizes
        all_periods, period_occurrences, period_locations = self._adaptive_window_analysis(
            counts, time_series, dt
        )

        # Filter and analyze detected periods
        filtered_periods = self._filter_and_analyze_periods(period_occurrences, period_locations)

        result = {
            'periods': filtered_periods
        }

        return result


class LombScarglePeriodDetector:
    def __init__(self, min_period=0.1, max_period=168, n_periods=5, time_unit='hours', samples_per_peak=5):
        self.min_period = min_period
        self.max_period = max_period
        self.n_periods = n_periods
        self.time_unit = time_unit
        self.samples_per_peak = samples_per_peak

    def process(self, activity_data):
        timestamps = np.array([point['time'] for point in activity_data])

        if len(timestamps) <= 1:
            return [], None

        t_min = timestamps.min()
        timestamps_normalized = timestamps - t_min
        timestamps_hours = timestamps_normalized / 3600

        ls = LombScargle(timestamps_hours, np.ones_like(timestamps_hours))
        min_freq = 1 / self.max_period
        max_freq = 1 / self.min_period

        frequency, power = ls.autopower(
            minimum_frequency=min_freq,
            maximum_frequency=max_freq,
            samples_per_peak=self.samples_per_peak
        )

        period = 1 / frequency
        peaks, properties = find_peaks(power, prominence=0.05 * np.max(power))
        peak_powers = power[peaks]
        peak_periods = period[peaks]
        sorted_indices = np.argsort(-peak_powers)
        peak_periods = peak_periods[sorted_indices]
        peak_powers = peak_powers[sorted_indices]

        if len(peak_periods) > self.n_periods:
            peak_periods = peak_periods[:self.n_periods]
            peak_powers = peak_powers[:self.n_periods]

        max_power = np.max(power)
        significances = peak_powers / max_power * 100

        detected_periods = []

        for i, (pd, pwr, sig) in enumerate(zip(peak_periods, peak_powers, significances)):
            if self.time_unit == 'minutes':
                period_value = pd * 60
            elif self.time_unit == 'seconds':
                period_value = pd * 3600
            else:
                period_value = pd

            detected_periods.append({
                'period': period_value,
                'power': float(pwr),
                'significance': float(sig)
            })

        return detected_periods, {
            'timestamps_hours': timestamps_hours,
            'period': period,
            'power': power
        }
