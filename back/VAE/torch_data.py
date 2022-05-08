import math
import numpy as np 
from torch.utils.data import Dataset
import pretty_midi
import pandas as pd

class BarTransform():

    def __init__(self, split, bars=1,note_count=60):
        self.split_size = bars*split
        self.note_count = note_count

    def get_sections(self, sample_length):
        return math.ceil(sample_length/ self.split_size)

    def __call__(self, sample):
        sample_length = len(sample)

        # Pad the sample with 0's if there's not enough to create equal splits into n bars
        leftover = sample_length % self.split_size
        if leftover != 0:
            padding_size = self.split_size - leftover
            padding = np.zeros((padding_size, self.note_count))
            sample = np.append(sample, padding, axis=0)


        sections = self.get_sections(sample_length)
        # Split into X equal sections
        split_list = np.array_split(sample, indices_or_sections=sections)


        return split_list

class MidiDataset(Dataset):
    """Pre-processed MIDI dataset."""

    def __init__(self, csv_file, transform, midi_start=0, midi_end=128,group_both_hands=True):
        """Init the mididataset object

        Args:
            csv_file (str): absolute path to the encoded piano_rolls csv file
            transform (object): transform object
            midi_start (int, optional): First pitch to be considered on a keyboard. Defaults to 0.
            midi_end (int, optional): Last pitch to be considered on a keyboard. Defaults to 0.. Defaults to 128.
            group_both_hands (bool, optional): Sum over same files to have an encoding of the whole music, 
            or consider each hand part as a different music piece. Defaults to True.
        """

        dtypes = {}
        column_names = [pretty_midi.note_number_to_name(n) for n in range(midi_start, midi_end)]
        for column in column_names:
            dtypes[column] = 'uint8'
        
        piano_rolls = pd.read_csv(csv_file, sep=';')
        piano_rolls.columns = ["piano_roll_name", "time_step"] + [pretty_midi.note_number_to_name(n) for n in range(midi_start, midi_end)]
        piano_rolls = piano_rolls.set_index(['piano_roll_name', 'time_step']).dropna().astype(dtypes)
        
        if group_both_hands:
            piano_rolls =  piano_rolls.reset_index()
            piano_rolls["file"] = piano_rolls["piano_roll_name"].apply(lambda x : x.split(":")[0])
            piano_rolls = piano_rolls.groupby(["file","time_step"], as_index=False).sum()
            piano_rolls = piano_rolls.rename(columns = {"file":"piano_roll_name"}).set_index(["piano_roll_name","time_step"])


        self.piano_rolls = piano_rolls

        self.transform = transform

        self.init_dataset()

    def init_dataset(self):
        """
            Sets up an array containing a pd index (the song name) and the song section,
            ie. [("Song Name:1", 0), ("Song Name:1", 1), ("Song Name:1", 2)]
            for use in indexing a specific section
        """
        indexer = self._get_indexer()

        self.index_mapper = []
        for i in indexer:
            split_count = self.transform.get_sections(len(self.piano_rolls.loc[i].values))
            for j in range(0, split_count):
                self.index_mapper.append((i, j))

    def __len__(self):
        return len(self.index_mapper)

    def get_mem_usage(self):
        """
            Returns the memory usage in MB
        """
        return self.piano_rolls.memory_usage(deep=True).sum() / 1024**2

    def _get_indexer(self):
        """
            Get an indexer that treats each first level index as a sample.
        """
        return self.piano_rolls.index.get_level_values(0).unique()

    def __getitem__(self, idx):
        """
            Our frame is multi-index, so we're thinking each song is a single sample,
            and getting the individual bars is a transform of that sample?
        """
        song_name, section = self.index_mapper[idx]

        # Add a column for silences
        piano_rolls = self.piano_rolls.loc[song_name].values
        silence_col = np.zeros((piano_rolls.shape[0], 1))
        piano_rolls_with_silences = np.append(piano_rolls, silence_col, axis=1)

        # Transform the sample (including padding)
        sample = piano_rolls_with_silences.astype('float')
        sample = self.transform(sample)[section]

        # Fill in 1's for the silent rows
        empty_rows = ~sample.any(axis=1)
        if len(sample[empty_rows]) > 0:
            sample[empty_rows, -1] = 1.

        sample = {'piano_rolls': sample}

        return sample
