import numpy as np
import os
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from back.SBGM.sbgm_fct import diffusion_coeff_fn, loss_fn, marginal_prob_std_fn
from back.SBGM.sampler import Euler_Maruyama_sampler
from back.VAE.torch_data import BarTransform, MidiDataset
from torch.optim import Adam
import torch
from back.SBGM.SBGM_models import ScoreNet
from torch.autograd import Variable
from structure import *
from back.SBGM.config import d
from back.MidiFile.config import d_encoding
from back.VAE.config import d_vae
import pandas as pd
from back.MidiFile.midi_file import MidiFileParser
from back.VAE.VAE import VariationalAutoencoder
from back.utils.utils import piano_roll_to_pretty_midi, threshold
import zipfile
import argparse
from base64 import b64encode

def generate(composer, n_output):
    ####### IMPORT PARAMETERS FROM CONFIG GILE
    notesperbar = d["notesperbar"]
    totalbars = d["totalbars"]
    batch_size = d["batch_size"]
    num_workers = d["num_workers"]
    test_split = d["test_split"]
    shuffle = d["shuffle"]
    group_both_hands = d["group_both_hands"]
    device = d["device"]
    drop_last = d["drop_last"]
    lr = d["lr"]
    n_epochs = d["n_epochs"]
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    ######### IMPORT DATA
    csv_path = f"{data_output_path}{composer}_encoded"
    print(csv_path)
    if not os.path.exists(f"{csv_path}.zip"):
        
        path_to_midi = f"{data_midi_path}{composer}/"
        logging_ = d_encoding["logging_"]
        path_to_output = d_encoding["path_to_output"]
        fs = d_encoding["fs"]
        transposer_ = d_encoding["transposer_"]
        chopster_ = d_encoding["chopster_"]
        trim_blanks_ = d_encoding["trim_blanks_"]
        minister_ = d_encoding["minister_"]
        arpster_ = d_encoding["arpster_"]
        cutster_ = d_encoding["cutster_"]
        padster_ = d_encoding["padster_"]

        mfp = MidiFileParser(path_to_midi, logging=logging_)
        mfp.get_piano_roll_df(path_to_output, 
                            fs, 
                            transposer_,
                            chopster_, 
                            trim_blanks_, 
                            minister_,
                            arpster_, 
                            cutster_, 
                            padster_)
    else: 
        with zipfile.ZipFile(f"{csv_path}.zip","r") as zip_ref:
            zip_ref.extractall(data_output_path)
        
    df = pd.read_csv(f"{csv_path}.csv", sep =';')

    #######LOAD DATA
    NUM_PITCHES= df.shape[1] - 2 + 1 
    TOTAL_NOTES=notesperbar*totalbars
    num_features=NUM_PITCHES

    transform = BarTransform(split=notesperbar,
                            bars=totalbars, 
                            note_count=NUM_PITCHES)
    midi_dataset = MidiDataset(csv_file=f"{csv_path}.csv", 
                            transform = transform, 
                            midi_start = 0,
                            midi_end = NUM_PITCHES - 1,
                            group_both_hands = group_both_hands) 

    data_loader = DataLoader(midi_dataset, 
                            shuffle=shuffle, 
                            batch_size=batch_size, 
                            num_workers=num_workers, 
                            drop_last=drop_last)

    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn))
    score_model = score_model.to(device)
    random_seed = 42
    np.random.seed(random_seed)


    #######VARIATIONNAL AUTOENCODE DATA
    path_to_VAE_pretrained_model = f'{composer}_VAE.pt'
    latent_features_VAE= d_vae["latent_features_VAE"]
    enc_hidden_size_VAE= d_vae["enc_hidden_size_VAE"]
    decoders_initial_size_VAE = d_vae["decoders_initial_size_VAE"]
    dropout_rate_VAE = d_vae["dropout_rate_VAE"]
    lr_VAE = d_vae["lr_VAE"]
    eps_i_VAE = d_vae["eps_i_VAE"]
    num_epochs_VAE= d_vae["num_epochs_VAE"]
    teacher_forcing_VAE = d_vae["teacher_forcing_VAE"]
    num_epochs_VAE = d_vae["num_epochs_VAE"]

    vae = VariationalAutoencoder(latent_features = latent_features_VAE, 
                                teacher_forcing = teacher_forcing_VAE, 
                                eps_i = eps_i_VAE, 
                                cuda = cuda, 
                                device = device,
                                enc_hidden_size = enc_hidden_size_VAE,
                                decoders_initial_size = decoders_initial_size_VAE,
                                dropout_rate = dropout_rate_VAE,
                                NUM_PITCHES=NUM_PITCHES,
                                totalbars = totalbars,
                                notesperbar = notesperbar).to(device)

    path_to_pretrained = f"{data_path}{path_to_VAE_pretrained_model}"
    if os.path.exists(path_to_pretrained):
        checkpoint = torch.load(path_to_pretrained)
        vae.load_state_dict(checkpoint)
    else: 
            
        dataset_size = len(midi_dataset)           
        test_size = int(test_split * dataset_size) 
        train_size = dataset_size - test_size      

        train_dataset, test_dataset = random_split(midi_dataset, [train_size, test_size])

        train_loader = DataLoader(train_dataset, 
                                shuffle=shuffle, 
                                batch_size=batch_size, 
                                num_workers=num_workers,
                                drop_last=True)
        test_loader = DataLoader(test_dataset, 
                                shuffle=shuffle, 
                                batch_size=batch_size, 
                                num_workers=num_workers, 
                                drop_last=True)

        optimizer = Adam(vae.parameters(), lr=lr_VAE)
        train_loss, train_kl, train_klw, valid_loss, valid_kl = vae.train_VAE(optimizer, 
                                                                                train_loader, 
                                                                                test_loader,
                                                                                path_to_pretrained, 
                                                                                num_epochs=num_epochs_VAE)


    #########REVERSE SDE LEARNING
    path_to_SBGM_pretrained = f"{composer}_SBGM.pth"
    if not os.path.exists(data_path + path_to_SBGM_pretrained):
        
        optimizer = Adam(score_model.parameters(), 
                        lr=lr)
        import tqdm
        tqdm_epoch = tqdm.notebook.trange(n_epochs)
        for epoch in tqdm_epoch:
            avg_loss = 0.
            num_items = 0
            for x in data_loader:
                x = Variable(x['piano_rolls'].type('torch.FloatTensor'))
                x = x.to(device)
                vae.set_scheduled_sampling(1.)
                outputs = vae(x,notesperbar,totalbars)
                z_latent = outputs['z'].detach()   
                z_latent = torch.reshape(z_latent, (batch_size, 1, totalbars*notesperbar, 32))
                loss = loss_fn(score_model, z_latent, marginal_prob_std_fn)
                optimizer.zero_grad()
                loss.backward()    
                optimizer.step()
                avg_loss += loss.item() * z_latent.shape[0]
                num_items += z_latent.shape[0]
            tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
            torch.save(score_model.state_dict(), data_path + path_to_SBGM_pretrained)

    else: 
        ckpt = torch.load(data_path + path_to_SBGM_pretrained, map_location=device)


    ########SAMPLING
    num_steps =  500 
    sample_batch_size = 64 
    score_model.load_state_dict(ckpt)
    samples = Euler_Maruyama_sampler(score_model, 
                    marginal_prob_std_fn,
                    diffusion_coeff_fn, 
                    batch_size = sample_batch_size, 
                    num_steps = num_steps,
                    device=device)
    samples = torch.squeeze(samples)

    #######DECODE
    x = next(iter(data_loader))
    x = Variable(x['piano_rolls'].type('torch.FloatTensor'))
    l_notes_gen = [vae.decode_VAE(
        x[i:i+1], 
        samples[i:i+1],
        gen_batch=1, 
        NUM_PITCHES = 
        NUM_PITCHES,
        totalbars = totalbars, 
        notesperbar = notesperbar) for i in np.random.choice([j for j in range(x.shape[0])], n_output)]

    l_out_midi_generated = [threshold(notes_gen[0,:,:].detach().cpu().numpy()) for notes_gen in l_notes_gen]

    ###### SAVE TO MIDI

    for i,out in enumerate(l_out_midi_generated): 
        piano_roll_to_pretty_midi(
            out[:,:-1],
            fs=8,
            to_mid = True, 
            filename = f"{data_midi_path}Final_generated_music/{composer}_{i}.mid"
            )
    l_final = []
    for i in range(n_output):
        mp4 = open(f'{data_midi_path}Final_generated_music/{composer}_{i}.mid','rb').read()
        l_final.append("data:video/mp4;base64," + b64encode(mp4).decode())

    return l_final


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.parse_args(description="Task API")
    parser.add_argument(
        "-c",
        "--composer",
        type=str,
        help="name of the composer",
        required=True,
    )
    parser.add_argument(
        "-n",
        "--number_output",
        type=int,
        help="number of output",
        required=True,
    )

    args = parser.parse_args()
    composer = args.composer
    n_output = args.n_output

    l_out_midi_generated = generate(composer, n_output)


