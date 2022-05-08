#@title debugging
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import softplus
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.kl import kl_divergence
from torch.autograd import Variable
from torch.nn.functional import binary_cross_entropy
from structure import data_path
import matplotlib.pyplot as plt
from IPython.display import Image, display, clear_output
import time
import os

from tqdm import tqdm

class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_features,teacher_forcing, eps_i,
                    cuda, device,  
                    enc_hidden_size = 256,
                    decoders_initial_size = 32, dropout_rate = 0.2,
                    NUM_PITCHES = 61, totalbars=16, notesperbar=16):
        """Init the VAE object

        Args:
            latent_features (int): dimension of the latent space
            teacher_forcing (boolean):param 
            eps_i (int): param
            cuda (string): machine 
            device (torch.device): device
            enc_hidden_size (int, optional): . Defaults to 256.
            decoders_initial_size (int, optional):  Defaults to 32.
            dropout_rate (float, optional): . Defaults to 0.2.
            NUM_PITCHES (int, optional): . Number of considered pitches.
            Often the number of keys kept on a keyboard + one for a silence-dedicated column
            Defaults to 61.
            totalbars (int, optional): Number of bars considered as a sample Defaults to 16.
            notesperbar (int, optional): Number of notes in one bar Defaults to 16.
        """

        super(VariationalAutoencoder, self).__init__()
        
        self.teacher_forcing = teacher_forcing
        self.eps_i = eps_i
        self.latent_features = latent_features
        self.cuda = cuda 
        self.device = device 
        self.enc_hidden_size = enc_hidden_size
        self.decoders_initial_size = decoders_initial_size
        self.totalbars = totalbars
        self.notesperbar = notesperbar
        self.NUM_PITCHES = NUM_PITCHES
        self.dropout_rate = dropout_rate

        #data goes into bidirectional encoder
        self.encoder = torch.nn.LSTM(
                batch_first = True,
                input_size = NUM_PITCHES,

                hidden_size = enc_hidden_size,
                num_layers = 1,
                bidirectional = True).to(device)
        
        #encoded data goes onto connect linear layer. inputs must be*2 because LSTM is bidirectional
        #output must be 2*latentspace because it needs to be split into miu and sigma right after.
        self.encoderOut = nn.Linear(in_features=enc_hidden_size*2, out_features=latent_features*2).to(device)
        
        #after being converted data goes through a fully connected layer
        self.linear_z = nn.Linear(in_features=latent_features, out_features=decoders_initial_size).to(device)
        
        self.dropout= nn.Dropout(p=dropout_rate).to(device)
        
        self.worddropout = nn.Dropout2d(p=dropout_rate).to(device)
        
        # Define the conductor and note decoder
        self.conductor = nn.LSTM(decoders_initial_size, decoders_initial_size, num_layers=1,batch_first=True).to(device)
        self.decoder = nn.LSTM(NUM_PITCHES+decoders_initial_size, decoders_initial_size, num_layers=1,batch_first=True).to(device)
        
        # Linear note to note type (classes/pitches)
        self.linear = nn.Linear(decoders_initial_size, NUM_PITCHES).to(device)

    def init_hidden(self, batch_size):
        #must be 2 x batch x hidden_size because its a bi-directional LSTM
        init = torch.zeros(2, batch_size, self.enc_hidden_size, device=self.device)
        c0 = torch.zeros(2, batch_size, self.enc_hidden_size, device=self.device)
    
        #2 because has 2 layers
        #n_layers_conductor
        init_conductor = torch.zeros(1, batch_size, self.decoders_initial_size, device=self.device)
        c_condunctor = torch.zeros(1, batch_size, self.decoders_initial_size, device=self.device)
        
        return init,c0,init_conductor,c_condunctor

    def use_teacher_forcing(self):
        with torch.no_grad():
            tf = np.random.rand(1)[0] <= self.eps_i
        return tf
    
    def set_scheduled_sampling(self, eps_i):
        self.eps_i = eps_i

    def forward(self, x,notesperbar,totalbars):

        batch_size = x.size(0)
        note = torch.zeros(batch_size, 1 , self.NUM_PITCHES,device=self.device)
        the_input = torch.cat([note,x],dim=1)
        
        outputs = {}
        
        #creates hidden layer values
        h0,c0,hconductor,cconductor = self.init_hidden(batch_size)
        
        x = self.worddropout(x)
        
        #resets encoder at the beginning of every batch and gives it x
        x, hidden = self.encoder.to(self.device)(x, ( h0,c0))
        
        #x=self.dropout(x)
        
        #goes from 4096 to 1024
        x = self.encoderOut.to(self.device)(x)      
        
        #x=self.dropout(x)
        
        # Split encoder outputs into a mean and variance vector 
        mu, log_var = torch.chunk(x, 2, dim=-1)
                
        # Make sure that the log variance is positive
        log_var = softplus(log_var)
               
        # :- Reparametrisation trick
        # a sample from N(mu, sigma) is mu + sigma * epsilon
        # where epsilon ~ N(0, 1)
                
        # Don't propagate gradients through randomness
        with torch.no_grad():
            batch_size = mu.size(0)
            epsilon = torch.randn(batch_size, 1, self.latent_features)
            

            if self.cuda:

                epsilon = epsilon.cuda()
        
        #setting sigma
        sigma = torch.exp(log_var*2)
        
        #generate z - latent space
        z = mu + epsilon * sigma
        
        #decrese space
        z = self.linear_z.to(self.device)(z)
        
        #z=self.dropout(z)
        
        #make dimensions fit (NOT SURE IF THIS IS ENTIRELY CORRECT)
        #z = z.permute(1,0,2)

        #DECODER ##############
        
        conductor_hidden = (hconductor,cconductor)
        
        counter=0
        

        notes = torch.zeros(batch_size,notesperbar*totalbars,self.NUM_PITCHES,device=self.device)

       
        note = torch.zeros(batch_size, 1 , self.NUM_PITCHES,device=self.device)

        
        # Go through each element in the latent sequence
        for i in range(totalbars):
            embedding, conductor_hidden = self.conductor.to(self.device)(z[:,i,:].view(batch_size,1, -1), conductor_hidden)    

            if self.use_teacher_forcing():
                
                decoder_hidden = (torch.randn(1,batch_size, self.decoders_initial_size,device=self.device), torch.randn(1,batch_size, self.decoders_initial_size,device=self.device))
                embedding = embedding.expand(batch_size, notesperbar, embedding.shape[2])
                e = torch.cat([embedding,the_input[:,range(i*notesperbar,i*notesperbar+notesperbar),:]],dim=-1)
                
                notes2, decoder_hidden = self.decoder(e, decoder_hidden)
                
                aux = self.linear(notes2)
                aux = torch.softmax(aux, dim=2);
                    
                notes[:,range(i*notesperbar,i*notesperbar+notesperbar),:]=aux;
            else:           
                 # Reset the decoder state of each 16 bar sequence

                decoder_hidden = (torch.randn(1,batch_size, self.decoders_initial_size,device=self.device), 
                torch.randn(1,batch_size, self.decoders_initial_size,device=self.device))

                
                for _ in range(notesperbar):
                    # Concat embedding with previous note
                    e = torch.cat([embedding, note], dim=-1)
                    e = e.view(batch_size, 1, -1)
                    # Generate a single note (for each batch)
                    note, decoder_hidden = self.decoder(e, decoder_hidden)
                    
                    aux = self.linear(note)
                    aux = torch.softmax(aux, dim=2);
                    
                    notes[:,counter,:]=aux.squeeze();
                    
                    note=aux
                    
                    counter=counter+1


        outputs["x_hat"] = notes
        outputs["z"] = z
        outputs["mu"] = mu
        outputs["log_var"] = log_var
        
        return outputs

    
    def lin_decay(self, i, train_loader, mineps=0):
        return np.max([mineps, 1 - (1/len(train_loader))*i])

    def inv_sigmoid_decay(i, rate=40):
        return rate/(rate + np.exp(i/rate))

    def ELBO_loss(cuda, y, t, mu, log_var, weight):
        cuda = torch.cuda.is_available()
        # Reconstruction error, log[p(x|z)]
        # Sum over features
        likelihood = -binary_cross_entropy(y, t, reduction="none")
        likelihood = likelihood.view(likelihood.size(0), -1).sum(1)

        # Regularization error: 
        # Kulback-Leibler divergence between approximate posterior, q(z|x)
        # and prior p(z) = N(z | mu, sigma*I).
        sigma = torch.exp(log_var*2)
        n_mu = torch.Tensor([0])
        n_sigma = torch.Tensor([1])
        if cuda:
            n_mu = n_mu.cuda()
            n_sigma = n_sigma.cuda()

        p = Normal(n_mu, n_sigma)
        q = Normal(mu, sigma)

        #The method signature is P and Q, but might need to be reversed to calculate divergence of Q with respect to P
        kl_div = kl_divergence(q, p)
        
        # In the case of the KL-divergence between diagonal covariance Gaussian and 
        # a standard Gaussian, an analytic solution exists. Using this excerts a lower
        # variance estimator of KL(q||p)
        #kl = -weight * torch.sum(1 + log_var - mu**2 - torch.exp(log_var), dim=(1,2))
        
        # Combining the two terms in the evidence lower bound objective (ELBO) 
        # mean over batch
        ELBO = torch.mean(likelihood) - (weight*torch.mean(kl_div)) # add a weight to the kl using warmup
        
        # notice minus sign as we want to maximise ELBO
        return -ELBO, kl_div.mean(),weight*kl_div.mean() # mean instead of sum

    def train_VAE(self, optimizer, train_loader, test_loader, PATH, 
                num_epochs = 100, warmup_epochs= 90, scheduled_decay_rate = 40,
                pre_warmup_epochs = 10, use_scheduled_sampling = False):
        """_summary_

        Args:
            optimizer (torch.optimizer): Adam
            train_loader (torch.dataloader): 
            test_loader (torch.datalaoder): 
            PATH (str): path to save checkpoints
            num_epochs (int, optional): number of epochs. Defaults to 100.
            warmup_epochs (int, optional): warmup eepochs. Defaults to 90.
            scheduled_decay_rate (int, optional): scheduled_decay_rate. Defaults to 40.
            pre_warmup_epochs (int, optional): pre_warmup_epochs. Defaults to 10.
            use_scheduled_sampling (bool, optional): use_scheduled_sampling. Defaults to False.

        Returns:
            tupple: losses
        """
        #optimizer = optim.Adam(net.parameters(), lr=0.001)
        tmp_img = data_path + "tmp_vae_out.png"
        warmup_lerp = 1/warmup_epochs
        warmup_w=0
        eps_i = 1
        train_loss, valid_loss = [], []
        train_kl, valid_kl,train_klw = [], [],[]
        start = time.time()
        print("Training epoch {}".format(0))
        #epochs loop
        for epoch in tqdm(range(num_epochs)):
            
            batch_loss, batch_kl,batch_klw = [], [],[]
            self.train()

            for i_batch, sample_batched in enumerate(train_loader):
                #if i_batch == 10:
                #    break
                x = sample_batched['piano_rolls']

                x = x.type('torch.FloatTensor')
                
                #if i_batch%10==0:
                #    print("batch:",i_batch)

                x = Variable(x)

                # This is an alternative way of putting
                # a tensor on the GPU
                x = x.to(self.device)
                
                ## Calc the sched sampling rate:
                if epoch >= pre_warmup_epochs and use_scheduled_sampling:
                    eps_i = self.inv_sigmoid_decay(i_batch, rate=scheduled_decay_rate)

                self.set_scheduled_sampling(eps_i)

                outputs = self.forward(x,notesperbar = self.notesperbar,totalbars = self.totalbars)
                x_hat = outputs['x_hat']
                mu, log_var = outputs['mu'], outputs['log_var']
  
                elbo, kl,kl_w = self.ELBO_loss(x_hat, x, mu, log_var, warmup_w)

                optimizer.zero_grad()
                elbo.backward()
                optimizer.step()

                batch_loss.append(elbo.item())
                batch_kl.append(kl.item())
                batch_klw.append(kl_w.item())


            train_loss.append(np.mean(batch_loss))
            train_kl.append(np.mean(batch_kl))
            train_klw.append(np.mean(batch_klw))
            torch.save(self.state_dict(),PATH)

            # Evaluate, do not propagate gradients
            with torch.no_grad():
                self.eval()

                # Just load a single batch from the test loader
                x = next(iter(test_loader))
                x = Variable(x['piano_rolls'].type('torch.FloatTensor'))

                x = x.to(self.device)

                self.set_scheduled_sampling(1.) # Please use teacher forcing for validations

                outputs = self.forward(x, notesperbar = self.notesperbar,totalbars = self.totalbars)
                x_hat = outputs['x_hat']
                mu, log_var = outputs['mu'], outputs['log_var']
                z = outputs["z"]
            
                elbo, kl,klw = self.ELBO_loss(x_hat, x, mu, log_var, warmup_w)

                # We save the latent variable and reconstruction for later use
                # we will need them on the CPU to plot
                x = x.to("cpu")
                x_hat = x_hat.to("cpu")
                z = z.detach().to("cpu").numpy()

                valid_loss.append(elbo.item())
                valid_kl.append(kl.item())
            
            if epoch >= pre_warmup_epochs:
                warmup_w = warmup_w + warmup_lerp
                if warmup_w > 1:
                    warmup_w=1.
            
            if epoch == 0:
                continue
                    
            # -- Plotting --
            f, axarr = plt.subplots(2, 1, figsize=(10, 10))
            
            
            # Loss
            ax = axarr[0]
            ax.set_title("ELBO")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Error')

            ax.plot(np.arange(epoch+1), train_loss, color="black")
            ax.plot(np.arange(epoch+1), valid_loss, color="gray", linestyle="--")
            ax.legend(['Training', 'Validation'])
            
            
            # KL / reconstruction
            ax = axarr[1]
            
            ax.set_title("Kullback-Leibler Divergence")
            ax.set_xlabel('Epoch')
            ax.set_ylabel('KL divergence')


            ax.plot(np.arange(epoch+1), train_kl, color="black")
            ax.plot(np.arange(epoch+1), valid_kl, color="gray", linestyle="--")
            ax.plot(np.arange(epoch+1), train_klw, color="blue", linestyle="--")
            ax.legend(['Training', 'Validation','Weighted'])
            
            print("Epoch: {}, {} seconds elapsed".format(epoch, time.time() - start))
            
            plt.savefig(tmp_img)
            plt.close(f)
            display(Image(filename=tmp_img))
            clear_output(wait=True)
            # os.remove(tmp_img)


        end_time = time.time() - start
        print("Finished. Time elapsed: {} seconds".format(end_time))
        return train_loss, train_kl, train_klw, valid_loss, valid_kl



     
    def decode_VAE(self, x, z_gen, gen_batch = 32,  
                    NUM_PITCHES = 61, totalbars = 16, 
                    decoders_initial_size = 32, notesperbar = 16):
        """Decode the data map in latent space

        Args:
            x (np.array): 
            z_gen (np.array): 
            gen_batch (int, optional): . Defaults to 32.
            NUM_PITCHES (int, optional): . Defaults to 61.
            totalbars (int, optional): . Defaults to 16.
            decoders_initial_size (int, optional): . Defaults to 32.
            notesperbar (int, optional): . Defaults to 16.
        """
        z_gen = z_gen.to(self.device)
        # Sample from latent space
        h_gen,c_gen,hconductor_gen,cconductor_gen = self.init_hidden(gen_batch)
        conductor_hidden_gen = (hconductor_gen,cconductor_gen)
        notes_gen = torch.zeros(gen_batch,notesperbar*totalbars,NUM_PITCHES,device=self.device)
        # For the first timestep the note is the embedding
        note_gen = torch.zeros(gen_batch, 1 , NUM_PITCHES,device=self.device)
        counter=0
        the_input = torch.cat([note_gen,x],dim=1)
        for i in range(totalbars):
            decoder_hidden_gen = (torch.randn(1,gen_batch, decoders_initial_size,device=self.device), torch.randn(1,gen_batch, decoders_initial_size,device=self.device))
            embedding_gen, conductor_hidden_gen = self.conductor(z_gen[:,i,:].view(gen_batch,1, -1), conductor_hidden_gen)
            embedding_gen = embedding_gen.expand(gen_batch, notesperbar, embedding_gen.shape[2])
            e = torch.cat([embedding_gen,the_input[:,range(i*notesperbar,i*notesperbar+notesperbar),:]],dim=-1)
            notes2, decoder_hidden_gen = self.decoder(e, decoder_hidden_gen)
            aux = self.linear(notes2)
            aux = torch.softmax(aux, dim=2)
            notes_gen[:,range(i*notesperbar,i*notesperbar+notesperbar),:]=aux
        return(notes_gen) 