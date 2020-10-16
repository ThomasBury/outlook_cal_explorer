import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import datetime
import holidays
from scipy.sparse.csgraph import connected_components

plt.style.use('fivethirtyeight')


# %matplotlib inline

# function to find the duration of overlapping meeting
# to be fair, I only add up the duration end - start of the total overlapping interval 
# instead of adding up all the duration
def reductionFunction(data):
    # create a 2D graph of connectivity between date ranges
    start = data['Start Time'].values
    end = data['End Time'].values
    graph = (start <= end[:, None]) & (end >= start[:, None])

    # find connected components in this graph
    n_components, indices = connected_components(graph)

    # group the results by these connected components
    return data.groupby(indices).aggregate({'Start Time': 'min',
                                            'End Time': 'max'})


# If the subject is empty or NA, it's probably a block slot for convenience but not a genuine meeting
def drop_empty_subject_meetings(cal_df):
    mask = cal_df['Subject'].isnull().values
    cal_df = cal_df[~mask]
    mask = cal_df['Subject'].isin(['', ' '])
    return cal_df[~mask]


# convert dates using a lookup, much faster if many rows
def lookup(s, dt_format='%d-%m-%y'):
    """
    This is an extremely fast approach to datetime parsing.
    For large data, the same dates are often repeated. Rather than
    re-parse these, we store all unique dates, parse them, and
    use a lookup to convert all dates.
    For hh:mm:ss  %H:%M:%S
    """
    dates = {date: pd.to_datetime(date, format=dt_format) for date in s.unique()}
    return s.apply(lambda v: dates[v])


# We want the tech, geeks and experts have a capacity of 6 hours per day
# Let's see when this is achieved
def capacity(duration_per_day, w_days, tresh=1.5):
    more_than_50pct = len(duration_per_day[duration_per_day.Duration_hour > tresh])
    return print("Number of days with more than {0:0.2f} hours in meetings: {1:3d} on {2:3d} working "
                 "days ({3:2.0f} % of the working days)".format(tresh, more_than_50pct, w_days,
                                                                100 * more_than_50pct / w_days))


# Check if object is a list of strings
def is_list_of_strings(obj):
    # This if statement makes sure input is a list that is not empty
    if obj and isinstance(obj, list):
        return all(isinstance(s, str) for s in obj)
    else:
        return False


# Filter out some events like Out-of-office, "homeworking" all day events, etc.
def events_to_filter_out(cal_df, list_of_patterns):
    assert is_list_of_strings(list_of_patterns)
    kwstr = '|'.join(list_of_patterns)
    mask = cal_df['Subject'].str.contains(kwstr, regex=True, na=False)
    return mask


# capture the meetings related to the agile framework we are using (scrum for instance)
# ['retro', 'daily', 'plan', 'groom', 'grooming', 'review', 'sprint', 'planning']
def agile_framework(cal_df, agile_kwlist):
    cal_df['agile_frwk'] = False
    mask = events_to_filter_out(cal_df, agile_kwlist)
    cal_df.loc[mask, 'agile_frwk'] = True
    return cal_df


def simplify_subject(cal_df, mapping_dic):
    cal_df['Subject_cat'] = 'various'
    for key, list_of_patterns in mapping_dic.items():
        assert isinstance(list_of_patterns, list)
        kwstr = '|'.join(list_of_patterns)
        mask = cal_df['Subject'].str.contains(kwstr, regex=True, na=False)
        cal_df.loc[mask, 'Subject_cat'] = key
    return cal_df


def meetings_duration(wd_df, ma_window=5):
    # Compute fairly the time spent in meetings per day (take into account the overlaps)
    duration_df = wd_df.groupby(['Date']).apply(reductionFunction).reset_index('Date')
    duration_df['Duration'] = duration_df['End Time'] - duration_df['Start Time']

    # Duration in min, hour, moving average, total average
    duration_per_day = duration_df[['Date', 'Duration']].groupby('Date').sum()
    duration_per_day['Duration_min'] = duration_per_day['Duration'].dt.total_seconds() / 60
    duration_per_day['Duration_hour'] = duration_per_day['Duration'].dt.total_seconds() / 3600
    duration_per_day['Duration_ma'] = duration_per_day['Duration_hour'].rolling(window=ma_window).mean()
    duration_per_day['Duration_avg'] = duration_per_day['Duration_hour'].mean()
    return duration_per_day


def plot_duration(duration_df, inception_date, title="Hours spent in all meeting per day since "):
    fig_title = title + inception_date
    chart = duration_df[['Duration_hour', 'Duration_ma', 'Duration_avg']].plot(figsize=(10, 6), title=fig_title)
    plt.show()
    return chart


def plot_meeting_duration(wd_df, frequency='W', topics=True):
    assert frequency in ['W', 'M']
    date_str = 'Date_' + frequency
    if topics:
        keep_cols = [date_str, 'Duration', 'Subject_cat']
        grby_cols = [date_str, 'Subject_cat']
    else:
        keep_cols = [date_str, 'Duration']
        grby_cols = [date_str]

    Duration_gr = wd_df[keep_cols].groupby(grby_cols).sum().reset_index()

    Duration_gr['Duration_hour'] = Duration_gr.Duration.dt.total_seconds() / 3600
    plt.figure(figsize=(8, 4))
    if topics:
        g = sns.lineplot(x=date_str, y="Duration_hour", hue="Subject_cat", data=Duration_gr, alpha=0.75)
    else:
        g = sns.lineplot(x=date_str, y="Duration_hour", data=Duration_gr)
    g.set_ylabel('Hours per ' + frequency)
    g.set_xlabel('')
    g.set_title('Hours spent in meetings')
    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontweight='light'
    )
    plt.xlabel('')
    plt.show()
    return g


# the main thing
def main(cal_path, inception_date='2020-01-05', ma_window=5):
    # read the outlook csv
    cal_df = pd.read_csv(outlook_path, encoding='latin1')

    # subject in lowercase, more convenient to work with (for filtering)
    cal_df['Subject_raw'] = cal_df['Subject']
    cal_df['Subject'] = cal_df['Subject'].str.lower()

    # Flag if meetings are agile, training, etc. 
    # The mapping is too simple and there might be some overlaps
    # The best would be to format the meeting subject as Jan is doing
    simplify_dic = {
        'training': ['class', 'training', 'workshop', 'QA', 'network'],
        'team': ['r&d', 'ds', 'collaboration', 'lunch', 'placeholder', 'cross'],
        'agile': ['retro', 'daily', 'weekly', 'plan', 'groom', 'grooming', 'review',
                  'sprint', 'planning', 'reboot', 'blocker', 'scenario', 'ceremony',
                  'stand', 'standup', 'retro', 'retrospecitve', 'demo']
    }
    cal_df = simplify_subject(cal_df, simplify_dic)

    # Time start and end
    # Start date is the first occurence of the recurring meeting, not the actual date

    # Convert to datetime
    date_cols = [s for s in list(cal_df) if "Date" in s]
    for col in date_cols:
        cal_df[col] = lookup(cal_df[col], dt_format='%d-%m-%Y')
        # cal_df[col] = pd.to_datetime(cal_df[col], format='%d-%m-%Y')

    time_cols = [s for s in list(cal_df) if "Time" in s]
    for col in time_cols:
        cal_df[col] = lookup(cal_df[col], dt_format='%H:%M:%S')

    # Compute the meetings duration
    cal_df['Date'] = cal_df['Reminder Date']
    cal_df['Date_M'] = cal_df.Date.dt.to_period('M').dt.to_timestamp()
    cal_df['Date_W'] = cal_df.Date.dt.to_period('W').dt.to_timestamp()
    cal_df['Duration'] = cal_df['End Time'] - cal_df['Start Time']

    # Select from the 6th of January 2020
    cal_df = cal_df[cal_df['Date'] > inception_date]  # datetime.date(2020,1,5)

    # Mask the out-of-office (I'm used to label them OoO for convenience), just guessing what could it be for you
    ooo_labels = ['OoO', 'OOO', 'ooo', 'congé', "holidays"]
    mask = events_to_filter_out(cal_df, ooo_labels)
    cal_df = cal_df[~mask]
    # Filter out the canceled events
    cancel_labels = ["cancel", "Cancel", "canceled", "Canceled", "homeworking", "Homeworking", "teleworking",
                     "Teleworking"]
    mask = events_to_filter_out(cal_df, cancel_labels)
    cal_df = cal_df[~mask]

    # Filter out the homeworking events
    home_labels = ["homeworking", "Homeworking", "teleworking", "Teleworking", "booze"]
    mask = events_to_filter_out(cal_df, home_labels)
    cal_df = cal_df[~mask]

    # counting the number of working days
    start = cal_df['Date'].dt.date.min()
    end = cal_df['Date'].dt.date.max()
    hol_days = holidays.BE()[start:end]
    w_days = np.busday_count(start, end, holidays=hol_days)

    # merging to have the missing working days (those without meetings, therefore not exported by outlook)
    wd_df = pd.DataFrame({'Date': pd.bdate_range(start=cal_df['Date'].min(), end=cal_df['Date'].max(),
                                                 holidays=hol_days, freq='C', weekmask="Mon Tue Wed Thu Fri")})
    wd_df = wd_df.set_index('Date').join(cal_df.set_index('Date'), how='left')  # .reset_index()
    wd_df['Duration'] = wd_df['Duration'].fillna(pd.Timedelta(seconds=0))

    duration_per_day_all_meetings = meetings_duration(wd_df, ma_window=5)
    agile_mask = wd_df['Subject_cat'] == 'agile'
    duration_per_day_agile_meetings = meetings_duration(wd_df[agile_mask], ma_window=5)

    # How many days we have more than X hours spent in meetings
    print("=" * 50 + '{:^25}'.format("All meetings") + "=" * 50)
    for treshold in np.arange(0.5, 8, 0.5):
        capacity(duration_per_day_all_meetings, w_days, tresh=treshold)

    print("=" * 50 + '{:^25}'.format("Agile related meetings") + "=" * 50)
    for treshold in np.arange(0.5, 8, 0.5):
        capacity(duration_per_day_agile_meetings, w_days, tresh=treshold)

    # Plotting the raw data, the ma and the overall avg
    chart_all_meetings = plot_duration(duration_per_day_all_meetings, inception_date)
    chart_only_agile = plot_duration(duration_per_day_agile_meetings, inception_date,
                                     title="Hours spent in agile meeting per day since ")

    print('\n\n\n')
    print("=" * 50 + '{:^40}'.format("Time spent in meetings - overall") + "=" * 50)
    plot_meeting_duration(wd_df, frequency='M', topics=False)
    print('\n\n\n')
    print("=" * 50 + '{:^40}'.format("Time spent in meetings - per topic") + "=" * 50)
    plot_meeting_duration(wd_df, frequency='M', topics=True)

    return wd_df, duration_per_day_all_meetings, duration_per_day_agile_meetings, chart_all_meetings, chart_only_agile


outlook_path = "C:/Users/cal_export/cal_2020.CSV"

if __name__ == '__main__':
    # out, raw = main(get_mushroom_data, 'Mushroom')
    # print(out.sort_values(by=['Dataset', 'Avg. Score']))

    wd_df, duration_per_day_all_meetings, duration_per_day_agile_meetings, chart_all_meetings, chart_only_agile = main(
        outlook_path)
