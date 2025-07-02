import pandas as pd
import numpy as np

from textstat import textstat

def text_length_score(X: pd.Series, min_len=10, max_len=500):
    """
    Calculate the text length score for each text in a pandas series.
    """
    unreadable_mask = X.apply(lambda x: len(x) < min_len or len(x) > max_len)
    score = 1 - (unreadable_mask.sum().sum() / len(X))
    return float(score)

def text_readability_score(x: pd.Series, min_score=30):
    """
    Calculate the text readability score for each text in a pandas series.
    """
    scores = x.apply(textstat.flesch_reading_ease)
    score = 1 - (scores < min_score).sum().sum() / len(scores)
    return float(score)

def time_stamp_gap_rate(x: pd.Series, min_time_delta=pd.Timedelta(minutes=10)):
    """
    Calculate the time stamp non-sequential rate for each text in a pandas series.
    """
    time_str = x.str.extract(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})')
    time_str.sort_values(inplace=True, by=0)
    time_str = time_str.reset_index(drop=True)
    if len(time_str) < 2:
        return 0
    timestamps = time_str.apply(lambda x: pd.to_datetime(x, errors='coerce'))
    time_gaps = timestamps.diff().dropna()
    gap_rate = (time_gaps < min_time_delta).sum().sum() / len(time_gaps)
    return float(gap_rate)

def text_metric(x: pd.Series, min_len=10, max_len=500, min_score=30, min_time_delta=pd.Timedelta(minutes=10)):
    """
    Calculate the text length, readability, and time stamp non-sequential rate score for each text in a pandas series.
    """
    length_score = text_length_score(x, min_len, max_len)
    readability_score = text_readability_score(x, min_score)
    time_gap_rate = time_stamp_gap_rate(x, min_time_delta)
    return (length_score + readability_score + time_gap_rate) / 3

if __name__ == '__main__':
    df = pd.DataFrame({
        'text': ['2021-01-01 00:00:00 This is a sample text. 2023-01-01 00:00:00 This is another sample text.', '2021-01-01 00:01:00 This is a very long text that is not readable.', '2021-01-01 00:02:00 This is another sample text.']
    })

    print(time_stamp_gap_rate(df['text']))
