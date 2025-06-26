import pandas as pd

df= pd.read_csv("C:/Users/bruno/Downloads/new_chunk_results_60k.csv")

df.columns
print(df.shape)
print(df.columns)
df.head()

df.info()


# Combine 'track' and 'artist' to identify each unique song
df['song_id'] = df['track'] + " - " + df['artist']

# Chunks per song (each row is a chunk)
chunks_per_song = df.groupby('song_id').size()

# Sections per song (count unique sections per song)
sections_per_song = df.groupby('song_id')['section'].nunique()

# Display stats
print("Chunks per song - Mean:", chunks_per_song.mean(), " | Std:", chunks_per_song.std())
print("Sections per song - Mean:", sections_per_song.mean(), " | Std:", sections_per_song.std())

# Min/Max for chunks per song
min_chunks = chunks_per_song.min()
max_chunks = chunks_per_song.max()

# Min/Max for sections per song
min_sections = sections_per_song.min()
max_sections = sections_per_song.max()

# Print the results
print("Chunks per song - Min:", min_chunks, " | Max:", max_chunks)
print("Sections per song - Min:", min_sections, " | Max:", max_sections)


# Unique song count based on track + artist
df['song_id'] = df['track'] + " - " + df['artist']
unique_songs = df['song_id'].nunique()

# Chunk count is simply number of rows
num_chunks = len(df)

print(f"Unique songs: {unique_songs}")
print(f"Total chunks: {num_chunks}")

# VAD columns that must be present and non-null
vad_columns = [
    'valence_1', 'valence_2', 'valence_3',
    'arousal_1', 'arousal_2', 'arousal_3',
    'dominance_1', 'dominance_2', 'dominance_3'
]

# Check for missing VAD values
missing_vad = df[vad_columns].isnull().any(axis=1)
incomplete_chunks_df = df[missing_vad]

# Songs with missing any VAD values
songs_with_missing_vad = incomplete_chunks_df['song_id'].unique()
print(f"Number of songs with incomplete VAD scores: {len(songs_with_missing_vad)}")

# Filter out those songs entirely
df_filtered = df[~df['song_id'].isin(songs_with_missing_vad)]
len(df_filtered)

unique_songs_filtered = df_filtered['song_id'].nunique()
num_chunks_filtered = len(df_filtered)

print(f"Remaining songs after filtering: {unique_songs_filtered}")
print(f"Remaining chunks after filtering: {num_chunks_filtered}")

# How many VAD values are missing per column
missing_by_column = incomplete_chunks_df[vad_columns].isnull().sum()
print("Missing VAD values by column:\n", missing_by_column)

# Optionally check a few rows
incomplete_chunks_df[['track', 'artist'] + vad_columns].head(10)


vad_columns = [
    'valence_1', 'valence_2', 'valence_3',
    'arousal_1', 'arousal_2', 'arousal_3',
    'dominance_1', 'dominance_2', 'dominance_3'
]

# Check per chunk if all VADs are present
df['vad_complete'] = df[vad_columns].notnull().all(axis=1)

# Now, for each song: does it have **at least one** valid chunk?
songs_with_valid_chunk = df.groupby('song_id')['vad_complete'].any()

# Keep only songs that have at least one chunk with complete VAD scores
valid_song_ids = songs_with_valid_chunk[songs_with_valid_chunk].index

# Filter the dataframe
df_filtered = df[df['song_id'].isin(valid_song_ids)]

# Summary
print(f"Total songs before filtering: {df['song_id'].nunique()}")
print(f"Total songs after filtering: {df_filtered['song_id'].nunique()}")
print(f"Total chunks after filtering: {len(df_filtered)}")

# For each chunk: is everything VAD-related missing?
df['vad_missing_all'] = df[vad_columns].isnull().all(axis=1)

# Group by song: are ALL chunks missing all VADs?
songs_missing_all_vad = df.groupby('song_id')['vad_missing_all'].all()
songs_completely_missing_vad = songs_missing_all_vad[songs_missing_all_vad].index

print(f"Songs with ALL chunks missing ALL VAD values: {len(songs_completely_missing_vad)}")



####################################################################
####################################################################
####################################################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
#################
#################ANÁLISES RERUN 60K##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
#################
#################

len(df)
df.info()
df['dominance_3'].describe()

# Define VAD score columns
vad_cols = [
    'valence_1', 'valence_2', 'valence_3',
    'arousal_1', 'arousal_2', 'arousal_3',
    'dominance_1', 'dominance_2', 'dominance_3'
]

# 1. Chunks with any missing values (NaN)
missing_mask = df[vad_cols].isnull().any(axis=1)

# 2. Chunks with any invalid values (outside [0.00, 1.00])
invalid_range_mask = df[vad_cols].apply(lambda col: (col < 0) | (col > 1)).any(axis=1)

# Combine both conditions
rerun_mask = missing_mask | invalid_range_mask
df_to_rerun = df[rerun_mask].copy()

# Save for rerun
df_to_rerun.to_csv("C:/Users/bruno/Downloads/chunks_to_rerun.csv", index=False)
print(f"Chunks to rerun: {len(df_to_rerun)}")

print("Chunks with missing values:", missing_mask.sum())
print("Chunks with out-of-range values:", invalid_range_mask.sum())
print("Total (union):", rerun_mask.sum())

# Examples of values outside the range
for col in vad_cols:
    print(f"{col} max: {df[col].max()}, min: {df[col].min()}")
    
    
import os
from tqdm import tqdm

# Load the filtered file with only chunks needing rerun
df = pd.read_csv("C:/Users/bruno/Downloads/chunks_to_rerun.csv")
len(df)

save_folder = "C:/Users/bruno/Downloads/vad_rerun_batches/"
os.makedirs(save_folder, exist_ok=True)
batch_size = 1000

existing_batches = set([
    int(f.split("_")[-1].split(".")[0])
    for f in os.listdir(save_folder) if f.startswith("vad_rerun_batch_")
])

total_chunks = len(df)
total_batches = (total_chunks // batch_size) + (1 if total_chunks % batch_size != 0 else 0)

####################################################################
####################################################################
####################################################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
#################
#################ANÁLISES VAD 60##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
##################################
#################
#################

df.columns

import matplotlib.pyplot as plt
# VAD columns
vad_cols = [
    'valence_1', 'valence_2', 'valence_3',
    'arousal_1', 'arousal_2', 'arousal_3',
    'dominance_1', 'dominance_2', 'dominance_3'
]

# Convert to numeric
df[vad_cols] = df[vad_cols].apply(pd.to_numeric, errors='coerce')

# Song ID column
df["song_id"] = df["track"].str.strip().str.lower() + "___" + df["artist"].str.strip().str.lower()

### 🎯 DATA PROFILE BEFORE FILTERING
initial_chunks = len(df)
initial_songs = df["song_id"].nunique()

### 🎯 Filtering: keep chunks where at least one VAD is valid and in (0, 1]
invalid_row = (
    df[vad_cols].isna().all(axis=1) | 
    ((df[vad_cols] <= 0) | (df[vad_cols] > 1)).all(axis=1)
)
df_valid = df[~invalid_row].copy()

### 🎯 Count songs that were COMPLETELY discarded (all chunks invalid)
# Songs where ALL chunks are invalid
all_discarded_songs = df[invalid_row].groupby("song_id").size()
valid_song_ids = set(df_valid["song_id"])
discarded_song_ids = set(all_discarded_songs.index) - valid_song_ids
num_discarded_songs = len(discarded_song_ids)

### Compute medians for valid chunks
df_valid["valence_median"] = df_valid[["valence_1", "valence_2", "valence_3"]].median(axis=1)
df_valid["arousal_median"] = df_valid[["arousal_1", "arousal_2", "arousal_3"]].median(axis=1)
df_valid["dominance_median"] = df_valid[["dominance_1", "dominance_2", "dominance_3"]].median(axis=1)

### Compute medians for valid chunks
df_valid["valence_mean"] = df_valid[["valence_1", "valence_2", "valence_3"]].mean(axis=1)
df_valid["arousal_mean"] = df_valid[["arousal_1", "arousal_2", "arousal_3"]].mean(axis=1)
df_valid["dominance_mean"] = df_valid[["dominance_1", "dominance_2", "dominance_3"]].mean(axis=1)

### 🎯 DATA PROFILE AFTER FILTERING
remaining_chunks = len(df_valid)
remaining_songs = df_valid["song_id"].nunique()
discarded_chunks = initial_chunks - remaining_chunks

print("📊 DATA PROFILE:")
print(f"• Initial chunks: {initial_chunks}")
print(f"• Initial songs: {initial_songs}")
print(f"• Remaining valid chunks: {remaining_chunks}")
print(f"• Remaining valid songs: {remaining_songs}")
print(f"• Discarded chunks: {discarded_chunks}")
print(f"• Discarded songs (all chunks invalid): {num_discarded_songs}")

### 📈 BOX PLOTS (with color)
import numpy as np

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Colors: mean = blue, median = orange
color_mean = '#4c72b0'
color_median = '#dd8452'
color_list = [color_mean, color_median]

medianprops=dict(color='black', linewidth=2)

# Boxplot for Valence
valence_data = [df_valid["mean_valence"].dropna(), df_valid["valence_median"].dropna()]
bplot1 = axes[0].boxplot(valence_data, patch_artist=True, labels=["Mean", "Median"], medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bplot1['boxes'], color_list):
    patch.set_facecolor(color)
axes[0].set_title("Valence: Mean vs Median")
axes[0].set_ylim(0, 1.05)

# Boxplot for Arousal
arousal_data = [df_valid["mean_arousal"].dropna(), df_valid["arousal_median"].dropna()]
bplot2 = axes[1].boxplot(arousal_data, patch_artist=True, labels=["Mean", "Median"], medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bplot2['boxes'], color_list):
    patch.set_facecolor(color)
axes[1].set_title("Arousal: Mean vs Median")
axes[1].set_ylim(0, 1.05)

# Boxplot for Dominance
dominance_data = [df_valid["mean_dominance"].dropna(), df_valid["dominance_median"].dropna()]
bplot3 = axes[2].boxplot(dominance_data, patch_artist=True, labels=["Mean", "Median"], medianprops=dict(color='black', linewidth=2))
for patch, color in zip(bplot3['boxes'], color_list):
    patch.set_facecolor(color)
axes[2].set_title("Dominance: Mean vs Median")
axes[2].set_ylim(0, 1.05)

# Add a global legend
handles = [
    plt.Line2D([], [], color=color_mean, marker='s', linestyle='None', markersize=10, label='Mean'),
    plt.Line2D([], [], color=color_median, marker='s', linestyle='None', markersize=10, label='Median')
]
fig.legend(handles=handles, loc='upper center', ncol=2)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


####################
########KDE plots
####################
######################
########################

import seaborn as sns
# Set Seaborn style
sns.set(style="whitegrid")

# Create 1x3 subplot
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# KDE for Valence
sns.kdeplot(df_valid["valence_median"].dropna(), ax=axes[0], fill=True, color="#4c72b0", linewidth=2)
axes[0].set_title("Valence (Median)")
axes[0].set_xlim(0, 1)
axes[0].set_xlabel("Median Score")
axes[0].set_ylabel("Density")

# KDE for Arousal
sns.kdeplot(df_valid["arousal_median"].dropna(), ax=axes[1], fill=True, color="#dd8452", linewidth=2)
axes[1].set_title("Arousal (Median)")
axes[1].set_xlim(0, 1)
axes[1].set_xlabel("Median Score")
axes[1].set_ylabel("")

# KDE for Dominance
sns.kdeplot(df_valid["dominance_median"].dropna(), ax=axes[2], fill=True, color="#55a868", linewidth=2)
axes[2].set_title("Dominance (Median)")
axes[2].set_xlim(0, 1)
axes[2].set_xlabel("Median Score")
axes[2].set_ylabel("")

plt.tight_layout()
plt.show()

cols_to_check = {
    "Valence": ["mean_valence", "valence_median"],
    "Arousal": ["mean_arousal", "arousal_median"],
    "Dominance": ["mean_dominance", "dominance_median"]
}

print("📊 Distribution Statistics:\n")

for dim, cols in cols_to_check.items():
    print(f"--- {dim} ---")
    for col in cols:
        series = df_valid[col].dropna()
        print(f"{col} | Std: {series.std():.4f} | Mean: {series.mean():.4f} | Min: {series.min():.4f} | Max: {series.max():.4f}")
    print()
    
# Step 1: Set invalid VAD cells to NaN
for col in vad_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')  # ensure numeric
    df.loc[(df[col] <= 0.0) | (df[col] > 1.0), col] = pd.NA
    
# Step 2: Keep rows that have at least one valid VAD score
df_valid = df[df[vad_cols].notna().any(axis=1)].copy()

df_valid["valence_median"] = df_valid[["valence_1", "valence_2", "valence_3"]].median(axis=1)
df_valid["mean_valence"] = df_valid[["valence_1", "valence_2", "valence_3"]].mean(axis=1)

df_valid["arousal_median"] = df_valid[["arousal_1", "arousal_2", "arousal_3"]].median(axis=1)
df_valid["mean_arousal"] = df_valid[["arousal_1", "arousal_2", "arousal_3"]].mean(axis=1)

df_valid["dominance_median"] = df_valid[["dominance_1", "dominance_2", "dominance_3"]].median(axis=1)
df_valid["mean_dominance"] = df_valid[["dominance_1", "dominance_2", "dominance_3"]].mean(axis=1)

for col in ["mean_valence", "valence_median", "mean_arousal", "arousal_median", "mean_dominance", "dominance_median"]:
    print(f"{col} → Min: {df_valid[col].min():.4f}, Max: {df_valid[col].max():.4f}")
    
print("📊 Distribution Statistics:\n")

for dim, cols in cols_to_check.items():
    print(f"--- {dim} ---")
    for col in cols:
        series = df_valid[col].dropna()
        print(f"{col} | Std: {series.std():.4f} | Mean: {series.mean():.4f} | Min: {series.min():.4f} | Max: {series.max():.4f}")
    print()
    
df.columns

df_valid[["track", "artist", "section"]] = df[["track", "artist", "section"]]

#################################333
########################CHORUS ANALYSIS
####################################
####################################

# Make sure 'track', 'artist', 'section' are in df_valid (copied from df)
for col in ["track", "artist", "section"]:
    if col in df.columns and col not in df_valid.columns:
        df_valid[col] = df[col]

# Prepare song ID for grouping
df_valid["song_id"] = df_valid["track"].str.lower().str.strip() + "___" + df_valid["artist"].str.lower().str.strip()

# 1. ✅ GENERAL STATS
total_chunks = len(df_valid)
total_songs = df_valid["song_id"].nunique()

# 2. ✅ FILTER FOR CHORUS
df_chorus = df_valid[df_valid["section"].str.contains("chorus", case=False, na=False)]
chorus_chunks = len(df_chorus)
chorus_songs = df_chorus["song_id"].nunique()

# 3. ✅ Chorus medians
chorus_valence_median = df_chorus["valence_median"].median()
chorus_arousal_median = df_chorus["arousal_median"].median()
chorus_dominance_median = df_chorus["dominance_median"].median()

# Seaborn style
sns.set(style="whitegrid")

# Create subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

# === Valence ===
sns.kdeplot(df_valid["valence_median"].dropna(), ax=axes[0], fill=True, color="#4c72b0", linewidth=2, label="All Sections")
sns.kdeplot(df_chorus["valence_median"].dropna(), ax=axes[0], fill=False, color="black", linestyle="--", linewidth=2, label="Chorus")
axes[0].set_title("Valence (Median)")
axes[0].set_xlim(0, 1)
axes[0].set_xlabel("Median Score")
axes[0].set_ylabel("Density")

# === Arousal ===
sns.kdeplot(df_valid["arousal_median"].dropna(), ax=axes[1], fill=True, color="#dd8452", linewidth=2, label="All Sections")
sns.kdeplot(df_chorus["arousal_median"].dropna(), ax=axes[1], fill=False, color="black", linestyle="--", linewidth=2, label="Chorus")
axes[1].set_title("Arousal (Median)")
axes[1].set_xlim(0, 1)
axes[1].set_xlabel("Median Score")
axes[1].set_ylabel("")

# === Dominance ===
sns.kdeplot(df_valid["dominance_median"].dropna(), ax=axes[2], fill=True, color="#55a868", linewidth=2, label="All Sections")
sns.kdeplot(df_chorus["dominance_median"].dropna(), ax=axes[2], fill=False, color="black", linestyle="--", linewidth=2, label="Chorus")
axes[2].set_title("Dominance (Median)")
axes[2].set_xlim(0, 1)
axes[2].set_xlabel("Median Score")
axes[2].set_ylabel("")

# Add a legend (only to first plot for simplicity)
axes[0].legend()

plt.tight_layout()
plt.show()

# 5. ✅ Print counts
print("🎵 Total valid chunks:", total_chunks)
print("🎵 Total valid songs:", total_songs)
print("🎶 Chunks with chorus:", chorus_chunks)
print("🎶 Songs with chorus:", chorus_songs)
