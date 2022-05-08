import torch
d = {
    "composer": "Mozart",
    "notesperbar": 8,
    "totalbars" : 16,
    "batch_size" :  32,
    "num_workers" : 4,
    "test_split" : .2,  
    "shuffle" :True, 
    "group_both_hands" :True,
    "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    "lr":1e-4 ,
    "n_epochs":50,
    "drop_last":True   
}