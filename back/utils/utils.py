import pretty_midi
import librosa.display as display
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd
import pretty_midi
import matplotlib.pyplot as plt
import librosa.display as display


class MidiBuilder():
    """Build a MIDI from a piano roll sample"""

    def __init__(self, midi_start=48, midi_end=108):
        """
        Args:
            midi_start (int): The first midi note in the dataset
            midi_end (int): The last midi note in the dataset
        """
        self.dtypes = {'piano_roll_name': 'object', 'timestep': 'uint32'}
        self.column_names = [pretty_midi.note_number_to_name(n) for n in range(midi_start, midi_end)]
        for column in self.column_names:
            self.dtypes[column] = 'uint8'


    def midi_from_piano_roll(self, sample, tempo = 120):
        """
            We're taking some assumptions here to reconstruct the midi.
        """
        piano_roll = pd.DataFrame(sample, columns=self.column_names, dtype='uint8')

        program = 0
        velocity = int(100)
        bps = tempo / 60
        sps = bps * 4 # sixteenth notes per second

        # Create a PrettyMIDI object
        piano_midi = pretty_midi.PrettyMIDI()

        piano = pretty_midi.Instrument(program=program)
        # Iterate over note names, which will be converted to note number later
        for idx in piano_roll.index:
            for note_name in piano_roll.columns:
                #print(note_name)

                # Check if the note is activated at this timestep
                if piano_roll.iloc[idx][note_name] == 1.:
                    # Retrieve the MIDI note number for this note name
                    note_number = pretty_midi.note_name_to_number(note_name)

                    note_start = idx/sps # 0 if tempo = 60
                    note_end = (idx+1)/sps # 0.25

                    # Create a Note instance, starting according to the timestep * 16ths, ending one sixteenth later
                    # TODO: Smooth this a bit by using lookahead
                    note = pretty_midi.Note(
                        velocity=velocity, pitch=note_number, start=note_start, end=note_end)
                    # Add it to our instrument
                    piano.notes.append(note)
        # Add the instrument to the PrettyMIDI object
        piano_midi.instruments.append(piano)
        return piano_midi

        # Write out the MIDI data
        #piano_midi.write('name.mid')

    def plot_midi(self, midi_sample):
        display.specshow(midi_sample.get_piano_roll(), y_axis='cqt_note', cmap=plt.cm.hot)

    def play_midi(self, midi_sample):
        fs = 44100
        synth = midi_sample.synthesize(fs=fs)
        return [synth], fs

def piano_roll_to_pretty_midi(piano_roll, fs=32, program=0,to_mid=True, filename=None):
    '''Convert a Piano Roll array into a PrettyMidi object
     with a single instrument.
    Parameters
    ----------
    piano_roll : np.ndarray, shape=(128,frames), dtype=int
        Piano roll of one instrument
    fs : int
        Sampling frequency of the columns, i.e. each column is spaced apart
        by ``1./fs`` seconds.
    program : int
        The program number of the instrument.
    Returns
    -------
    midi_object : pretty_midi.PrettyMIDI
        A pretty_midi.PrettyMIDI class instance describing
        the piano roll.
    '''
    piano_roll = piano_roll.T
    notes, frames = piano_roll.shape
    pm = pretty_midi.PrettyMIDI()
    instrument = pretty_midi.Instrument(program=program)

    # pad 1 column of zeros so we can acknowledge inital and ending events
    piano_roll = np.pad(piano_roll, [(0, 0), (1, 1)], 'constant')

    # use changes in velocities to find note on / note off events
    velocity_changes = np.nonzero(np.diff(piano_roll).T)

    # keep track on velocities and note on times
    prev_velocities = np.zeros(notes, dtype=int)
    note_on_time = np.zeros(notes)

    for time, note in zip(*velocity_changes):
        # use time + 1 because of padding above
        velocity = piano_roll[note, time + 1]
        time = time / fs
        if velocity > 0:
            if prev_velocities[note] == 0:
                note_on_time[note] = time
                prev_velocities[note] = velocity
        else:
            pm_note = pretty_midi.Note(
                velocity=prev_velocities[note],
                pitch=note,
                start=note_on_time[note],
                end=time)
            instrument.notes.append(pm_note)
            prev_velocities[note] = 0
    pm.instruments.append(instrument)
    if to_mid:
        pm.write(filename)
    return pm

def threshold(x):
  rows, columns = x.shape
  out = np.empty((rows, columns))
  for i in range(rows): 
    var = np.sqrt(np.std(x[i,:]))
    
    for j in range(columns) :
      if x[i,j] > var:
        out[i,j] = 127
      else : 
        out[i,j] = 0
  return out
