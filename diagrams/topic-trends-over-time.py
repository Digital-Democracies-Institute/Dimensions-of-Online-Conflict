import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

dict_ordering = {
'("social distancing")':14,
'("will smith" slap)':13,
'(elon twitter)':12,
'(depp heard)':11,
'((canada truckers) OR (canadian truckers))':10,
'(libsoftiktok)':9,
'(biden loans)':8,
'("defund the police")':7,
'(metaverse)':6,
'("roe v wade")':5,
'(tigray ethiopia)':4,
'(vaccine)':3,
'("rogers outage")':2,
'(india pakistan missile)':1
}

dict_proper_name = {
    '("social distancing")':'Social Distancing',
'("will smith" slap)':'Will Smith Slap',
'(elon twitter)':'Elon Twitter',
'(depp heard)':'Depp Heard',
'((canada truckers) OR (canadian truckers))':'Canadian Truckers',
'(libsoftiktok)':'Libsoftiktok',
'(biden loans)':'Biden Loans',
'("defund the police")':'Defund the Police',
'(metaverse)':'Metaverse',
'("roe v wade")':'Roe v Wade',
'(tigray ethiopia)':'Tigray Ethiopia',
'(vaccine)':'Vaccine',
'("rogers outage")':'Rogers Outage',
'(india pakistan missile)':'India Pakistan Missile'
}

counts_df = pd.read_csv('counts_over_time.csv')

# Convert the "start" column to datetime format
counts_df['start'] = pd.to_datetime(counts_df['start'])

# Truncate query text before "is:reply"
counts_df['truncated_query'] = counts_df['query'].apply(lambda x: x.split("is:reply")[0].strip())
counts_df['proper_title'] = counts_df['truncated_query'].apply(lambda x: dict_proper_name[x] if x in dict_proper_name else '')

# Group by truncated query and find the peak count for each group
grouped_counts = counts_df.groupby('proper_title')

counts_df['ordering'] = counts_df['truncated_query'].apply(lambda x: dict_ordering[x] if x in dict_ordering else 999)
counts_df = counts_df[counts_df['proper_title'] != '']

counts_df = counts_df[['end', 'start', 'tweet_count', 'query',
       'proper_title', 'ordering']]

# Group by truncated query and find the peak count for each group
grouped_counts = counts_df.groupby('proper_title')
peak_counts = grouped_counts['ordering'].max().sort_values()



# Assuming counts_df is already defined and has columns "start", "query", "count"

# Convert the "start" column to datetime format
counts_df['start'] = pd.to_datetime(counts_df['start'])

# Truncate query text before "is:reply"
#counts_df['proper_title'] = counts_df['query'].apply(lambda x: x.split("is:reply")[0].strip())

# Group by truncated query and find the peak count for each group
grouped_counts = counts_df.groupby('proper_title')
peak_counts = grouped_counts['ordering'].max().sort_values()

# Create a figure and set its size
fig, ax = plt.subplots(figsize=(10, len(peak_counts) * 0.5))

# Iterate through queries in the order of peak counts
for i, (query, peak_count) in enumerate(peak_counts.items()):
    # Get the data for the current query
    data = counts_df[counts_df['proper_title'] == query].sort_values('start')
    
    # Normalize the counts to [0, 1] range
    norm_counts = (data['tweet_count'] - data['tweet_count'].min()) / (data['tweet_count'].max() - data['tweet_count'].min())

    # Plot the sparkline for the current query
    ax.plot(data['start'], norm_counts + i, linewidth=1)
    
    # Add a dot at the end of the sparkline
    ax.plot(data['start'].iloc[-1], norm_counts.iloc[-1] + i, marker='o', markersize=4, color='k')

# Set the y-axis labels to the queries
ax.set_yticks(np.arange(len(peak_counts)))
ax.set_yticklabels(peak_counts.index)

# Format the x-axis
ax.xaxis.set_major_locator(mdates.MonthLocator(interval=6))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.xticks(rotation=45)

# Remove the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Save the figure to a file
plt.savefig('sparklines_clean.pdf', bbox_inches='tight')

# Show the plot
plt.show()
