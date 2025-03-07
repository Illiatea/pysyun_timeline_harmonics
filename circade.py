class PeriodToCircadeText:
    SECONDS_IN_HOUR = 3600
    SECONDS_IN_DAY = 86400
    SECONDS_IN_WEEK = 604800
    SECONDS_IN_MONTH = 2628000  # Approximate (30.44 days)

    @classmethod
    def process(cls, periods):
        converted_data = []
        
        for item in periods:
            period_seconds = item['period'] / 1000  # Convert from milliseconds to seconds
            converted_data.append({
                'seconds': period_seconds,
                'hours': period_seconds / cls.SECONDS_IN_HOUR,
                'days': period_seconds / cls.SECONDS_IN_DAY,
                'weeks': period_seconds / cls.SECONDS_IN_WEEK,
                'months': period_seconds / cls.SECONDS_IN_MONTH,
                'volume': len(item['value'])
            })
        
        return converted_data
