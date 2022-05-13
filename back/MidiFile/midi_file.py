import os
import sys
import pretty_midi
import logging
import pandas as pd
import sys
from tqdm import tqdm
from back.preprocessing.encoding import *



class MidiFileParser():
    def __init__(self, src,instrument=None,program=None, logging=False):
        
        """_summary_

        Args:
            src (_type_): _description_
        """
        self.src = src
        self.instrument = instrument
        self.program = program
        self.logging = logging

    @property
    def get_instrument_df(self):
        """_summary_

        Returns:
            _type_: _description_
        """
        instrument_ary = [[]]
        instrument_ary.append(['program', 'is_drum', 'name','filepath'])
        midi_files=list(sorted(os.listdir(self.src)))
        for index, file in enumerate(midi_files):
            if self.logging: 
                logging.info("{}/{}: Loading and parsing {}".format(index, len(midi_files), os.path.basename(file)))
            try:
                pm = pretty_midi.PrettyMIDI(file)
                instruments = pm.instruments
                for instrument in instruments:
                    instrument_ary.append([instrument.program, instrument.is_drum, instrument.name.replace(';',''),file])
            except:
                continue
        return pd.DataFrame(data=instrument_ary, columns=["program", "is_drum", "name", "filepath"]).dropna(subset=['name'])
    
    def get_instruments_object(self, filename):
        """_summary_

        Args:
            filename (_type_): _description_

        Returns:
            _type_: _description_
        """
        pm = pretty_midi.PrettyMIDI(filename)
        instruments = pm.instruments
        return instruments
                
    def encoding(self, filename, fs):

        semi_shift = transposer(filename)
        pm = pretty_midi.PrettyMIDI(filename)
        # sampling_freq = 1/ (pm.get_beats()[1]/4)
        sampling_freq = 1/ fs
        l = []
        for j, instrument in enumerate(pm.instruments):
            if instrument.program == 0 and self.instrument in instrument.name.lower():
                for note in instrument.notes:
                    note.pitch += semi_shift

                df = encode_dummies(instrument, sampling_freq).fillna(value=0) 
                df.reset_index(inplace=True, drop=True)
                top_level_index = "{}_{}:{}".format(filename.split("/")[-1], 0, j)
                df['timestep'] = df.index
                df['piano_roll_name'] = top_level_index
                df = df.set_index(['piano_roll_name', 'timestep'])
                l.append(df)
        return pd.concat(l)

    def get_piano_roll_df(self,path_to_csv, fs,transposer_=True, chopster_=False, trim_blanks_ = False, minister_=False,arpster_=False, cutster_=False, padster_=False):
        """
        """
        if self.logging:
            logging.basicConfig(filename=f'{path_to_csv}midiparser.log', level=logging.DEBUG)
            logging.info("*****parsing all files in {} with {} playing***********".format(self.src, self.instrument))

        midi_files=[self.src + x for x in list(sorted(os.listdir(self.src)))]
        l=[]
        logging.info("******ENCODING*********")
        for i, file in tqdm(enumerate(midi_files)):
            song_name = os.path.basename(file)  
            if transposer_:
                semi_shift = transposer(file)
            else : 
                semi_shift = 0
            try:
                pm = pretty_midi.PrettyMIDI(file)
            except Exception as e:
                logging.warning("{}/{}: {}. ENCOUNTERED EXCEPTION {}".format(i, len(midi_files), song_name,str(e)))
                print("{}/{}: {}. ENCOUNTERED EXCEPTION {}".format(i, len(midi_files), song_name,str(e)))
                continue

            instruments = pm.instruments

            for j,instrument in enumerate(instruments):
                for note in instrument.notes:
                    note.pitch += semi_shift
                try:
                    df = encode_dummies(instrument, fs).fillna(value=0) 
                except Exception as e:
                    logging.warning("{}/{}: {}. ENCOUNTERED EXCEPTION {}".format(i, len(midi_files), song_name,str(e)))
                    print("{}/{}: {}. ENCOUNTERED EXCEPTION {}".format(i, len(midi_files), song_name,str(e)))
                    continue
                if df is None:
                    logging.warning("{}/{}: {}. IS AN EMPTY TRACK".format(i, len(midi_files), song_name))
                    print("{}/{}: {}. IS AN EMPTY TRACK".format(i, len(midi_files), song_name))
                    continue
                if chopster_:
                    df = chopster(df)
                if trim_blanks_:                 
                    df = trim_blanks(df)
                if minister_:
                    df = minister(df)   
                if arpster_:         
                    df = arpster(df)
                if padster_: 
                    df = padster(df)
                if cutster_: 
                    df = cutster(df)

                df.reset_index(inplace=True, drop=True)
                top_level_index = "{}_{}:{}".format(song_name, i, j)
                df['timestep'] = df.index
                df['piano_roll_name'] = top_level_index
                df = df.set_index(['piano_roll_name', 'timestep'])
                l.append(df)
                if self.logging:
                    logging.info("{}:{}/{}: {}. ENCODED SUCCESSFULLY".format(i, j,len(midi_files), song_name))
    
        df = pd.concat(l)
        df.to_csv(f"{path_to_csv}encoded.csv", sep=';', encoding='utf-8', header=False)

                

