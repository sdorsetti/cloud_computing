import pandas as pd
import pretty_midi
import numpy as np
import music21

def encode_dummies(instrument, sampling_freq):
    """_summary_

    Args:
        instrument (_type_): _description_

    Returns:
        _type_: _description_
    """

    note_columns = [pretty_midi.note_number_to_name(n) for n in range(0,128)]
    pr = instrument.get_piano_roll(fs=sampling_freq).astype('uint8').T
    return pd.DataFrame(pr, columns=note_columns)


def trim_blanks(df):
    """_summary_

    Args:
        df (_type_): _description_

    Returns:
        _type_: _description_
    """
    nonzero = df.apply(lambda s: s != 0)
    first_nonzero = df[nonzero].apply(pd.Series.first_valid_index).min()
    return df.iloc[int(first_nonzero):]

def chopster(dframe):    
    dframe.drop(labels=[pretty_midi.note_number_to_name(n) for n in range(108,128)], axis=1, inplace=True)
    dframe.drop(labels=[pretty_midi.note_number_to_name(n) for n in range(0,48)], axis=1, inplace=True)
    return dframe

def minister(dframe):

    return dframe.where(dframe<1, 1)

def arpster(dframe):
    # Count amount of notes being played at once.
    note_amount = np.asarray(dframe.astype(bool).sum(axis=1))
    i = 0
    
    # Slide through whole MIDI
    while i < dframe.shape[0]:
        # Check if note is single
        if note_amount[i] == 1:
            i += 1
            continue
        elif note_amount[i] > 1: 
            hits = 0
            hit_index = []
            # Calculates the amount of notes being played
            for j in range(dframe.shape[1]):
                if dframe.iloc[i,j] == 1:
                    hit_index.append(j)
                    hits += 1
                    if hits == note_amount[i]:
                        break

            length = 0
            
            # Removes all notes such that chords are turned into arpeggios.
            # Ensures that all values in hit_index are the same as ones in 
            # dframe row.
            while False not in (dframe.iloc[i+length, hit_index] == 1).values:
                for k in range(len(hit_index)):
                    if k != (length % hits):
                        dframe.iloc[i+length, hit_index[k]] = 0
                length += 1
                if len(note_amount) <= i+length or note_amount[i+length-1] != note_amount[i+length]:
                    break

            # Skip ahead to next note
            i += length
                
        # Maybe a case where we count ithe amount of silent steps going ahead
        elif note_amount[i] == 0:
            i += 1
            continue
        
    return dframe

def cutster(dframe, frame_size, undesired_silence):
    # Chop up if the window size fits the music
    
    # Check if frame size is greater than MIDI length
    # Pad with zeros
    if frame_size > dframe.shape[0]/16:
        return dframe
    
    note_amount = np.asarray(dframe.astype(bool).sum(axis=1))
    zero_amount = 0

    i = 0
    while i < len(note_amount):
        # Cuts out silent measures if greater than undesired_silence
        if zero_amount/16 > undesired_silence and note_amount[i] != 0:
            drop_amount = [j for j in range(i-zero_amount,i)]
            dframe.drop(drop_amount, inplace=True)
            note_amount = np.delete(note_amount, drop_amount)
            i -= zero_amount-1
            zero_amount = 0
            
        elif note_amount[i] != 0:
            if zero_amount != 0:
                zero_amount = 0
            i += 1
        # Count sequential zeros
        elif note_amount[i] == 0:
            zero_amount += 1
            i += 1
        
    return dframe

def padster(dframe):
    return dframe.fillna(0)

def transposer(midi_file):
    # converting everything into the key of C major or A minor
    # Major conversion
    majors = dict([("A-", 4),("G#", 4),("A", 3),("A#", 2),("B-", 2),("B", 1),("C", 0),("C#", -1),("D-", -1),("D", -2),("D#", -3),("E-", -3),("E", -4),("F", -5),("F#", 6),("G-", 6),("G", 5)])
    # Minor conversion
    minors = dict([("G#", 1), ("A-", 1),("A", 0),("A#", -1),("B-", -1),("B", -2),("C", -3),("C#", -4),("D-", -4),("D", -5),("D#", 6),("E-", 6),("E", 5),("F", 4),("F#", 3),("G-", 3),("G", 2)])

    score = music21.converter.parse(midi_file)
    key = score.analyze('key')
    if key.mode == "major":
        halfSteps = majors[key.tonic.name]

    elif key.mode == "minor":
        halfSteps = minors[key.tonic.name]

    return halfSteps