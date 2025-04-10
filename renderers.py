import matplotlib.pyplot as plt
import numpy as np
import math


def gaussian_function(x, mean, std):
    """
    Create a Gaussian (normal) distribution function.
    """
    return np.exp(-0.5 * ((x - mean) / std) ** 2)


class StackedAreaHarmonicsChart:
    def __init__(self):
        pass

    def process(self, data):
        """
        Create a stacked area chart visualization where each period
        is represented as a colored area.

        Parameters:
        data (list): A list of groups, where each group is a list of dictionaries
                    with 'days' and 'power_spectrum' keys.

        Returns:
        matplotlib.figure.Figure: The figure containing the visualization
        """
        # Process the data - sort each group by days (small to large)
        processed_data = []
        for group in data:
            sorted_group = sorted(group, key=lambda x: x['days'])  # Змінено на сортування за зростанням
            processed_data.append(sorted_group)

        # Create a figure for each group
        figs = []

        for group_idx, group in enumerate(processed_data):
            # Create figure
            fig, ax = plt.subplots(figsize=(15, 6))

            # Find maximum days for x-axis
            max_days = max(period['days'] for period in group)

            # Create x-axis range
            x = np.linspace(0, max_days, 1000)

            # Get a list of distinct colors
            colors = plt.cm.viridis(np.linspace(0, 1, len(group)))

            # Store y values for stacking
            y_stack = np.zeros_like(x)

            # Create area for each period
            for period_idx, period in enumerate(group):
                days = period['days']
                power = period['power_spectrum']  # Замінено volume на power_spectrum

                # Create a bell curve for this period
                # Center it around days/2 with width proportional to days
                mean = days / 2
                std = days / 6
                y = gaussian_function(x, mean, std)
                y = y / y.max() * power  # Масштабувати за power_spectrum

                # Stack on top of previous
                ax.fill_between(x, y_stack, y_stack + y,
                                color=colors[period_idx],
                                alpha=0.7,
                                label=f'Period {period_idx + 1}: {days:.1f} days, Power: {power:.2f}')  # Додано інформацію про потужність

                # Update stack
                y_stack += y

            # Set title and labels
            ax.set_title(f'Group {group_idx + 1} - Harmonic Power Spectrum', fontsize=16, fontweight='bold')
            ax.set_xlabel('Days', fontsize=12)
            ax.set_ylabel('Power Spectrum', fontsize=12)  # Змінено з 'Intensity' на 'Power Spectrum'

            # Set axis limits
            ax.set_xlim(0, max_days)
            ax.set_ylim(0, None)

            # Add grid and legend
            ax.grid(linestyle='--', alpha=0.3)
            ax.legend(loc='upper right')

            # Store the figure
            figs.append(fig)

        return figs


class WeeklyHarmonicsChart:
    def __init__(self):
        # Reduce the default font size for all annotations
        plt.rc('font', size=8)

    def process(self, data):
        """
        Create a visualization of periods (in days) as squares organized into weekly columns.

        Parameters:
            data (list): A list of groups, where each group is a list of dicts
                         with 'days' and 'power_spectrum' keys.

        Returns:
            matplotlib.figure.Figure: The figure containing all groups.
        """
        # Close any existing figures to avoid overlap
        plt.close('all')

        # Sort each group by the number of days
        processed = [sorted(group, key=lambda x: x['days']) for group in data]

        # Determine the global maximum power_spectrum for a single shared colorbar
        max_power_global = max(
            period['power_spectrum']
            for group in processed
            for period in group
        )

        # Prepare figure and subplots
        fig, axes = plt.subplots(
            len(processed), 1,
            figsize=(15, 3 * len(processed)),
            gridspec_kw={'hspace': 0.5}
        )
        if len(processed) == 1:
            axes = [axes]

        # Enable constrained layout for better spacing
        fig.set_constrained_layout(True)

        square_size = 0.8
        cmap = plt.cm.viridis

        for gi, group in enumerate(processed):
            ax = axes[gi]
            ax.set_frame_on(False)
            ax.set_title(f'Group {gi+1} – Weekly Harmonic Pattern',
                         fontsize=12, fontweight='bold', pad=12)

            current_x = 0
            max_height = 0

            for pi, period in enumerate(group):
                days = period['days']
                power = period['power_spectrum']

                # Calculate how many weeks (columns) are needed
                weeks = math.ceil(days / 7)

                # Prepare the label text with appropriate units
                pd = power
                suffix = ''
                if power >= 1e9:
                    pd, suffix = power/1e9, 'B'
                elif power >= 1e6:
                    pd, suffix = power/1e6, 'M'
                elif power >= 1e3:
                    pd, suffix = power/1e3, 'K'
                label = f'{days:.1f}d, {pd:.2f}{suffix}'

                # Alternate vertical position of labels to reduce overlap
                label_y = -0.5 - (pi % 2) * 1.0
                ax.text(
                    current_x + weeks/2, label_y, label,
                    fontsize=8, va='top', ha='center',
                    bbox=dict(facecolor='white', alpha=0.9, pad=1.2,
                              boxstyle='round,pad=0.2', edgecolor='lightgray')
                )

                # Determine color based on normalized power
                norm_power = power / max_power_global if max_power_global > 0 else 0
                color = cmap(norm_power)

                # Draw one square per day
                for d in range(int(days)):
                    week = d // 7
                    dow = d % 7
                    x = current_x + week
                    y = dow + 1
                    rect = plt.Rectangle(
                        (x, y), square_size, square_size,
                        facecolor=color, edgecolor='white', linewidth=0.5, alpha=0.9
                    )
                    ax.add_patch(rect)

                # Keep track of the tallest column (7 days + label space)
                max_height = max(max_height, 8)

                # Add dynamic horizontal padding based on label length
                padding = max(1.0, len(label) / 10.0)
                current_x += weeks + padding

            # Configure axes limits and grid lines
            ax.set_xlim(-0.5, current_x)
            ax.set_ylim(-2.5, max_height + 0.5)
            ax.set_xticks(np.arange(0, current_x, 1))
            ax.set_xticklabels([])
            ax.grid(axis='x', linestyle='--', alpha=0.3)
            ax.set_yticks([])

        # Create a shared colorbar for all subplots
        sm = plt.cm.ScalarMappable(cmap=cmap,
                                   norm=plt.Normalize(0, max_power_global))
        sm.set_array([])
        cbar = fig.colorbar(
            sm, ax=axes, orientation='horizontal',
            pad=0.02, fraction=0.05
        )

        # Label the colorbar with the appropriate scale
        if max_power_global >= 1e9:
            scale, lbl = 1e9, 'Power Spectrum (billions)'
        elif max_power_global >= 1e6:
            scale, lbl = 1e6, 'Power Spectrum (millions)'
        elif max_power_global >= 1e3:
            scale, lbl = 1e3, 'Power Spectrum (thousands)'
        else:
            scale, lbl = 1, 'Power Spectrum'
        cbar.set_label(lbl)
        cbar.ax.set_xticklabels([f'{t/scale:.1f}' for t in cbar.get_ticks()])

        return fig
