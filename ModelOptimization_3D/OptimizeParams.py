# Function to wrap around ConvVAE_func.py
# Scans parameter ranges (grid search approach)

from ConvVAE_func_3D import Conv_VAE3D


# Settings to evaluate
n_epochs_range = [100]
batch_size_range = [64]
learning_rate_range = [0.001,0.002]
decay_rate_range = [0.05,0.1,0.15,0.2]
latent_dim_range = [4]


for n_epochs in n_epochs_range:
    for batch_size in batch_size_range:
        for learning_rate in learning_rate_range:
            for decay_rate in decay_rate_range:
                for latent_dim in latent_dim_range:
                    # IMPORTANT:
                    # OUTPUT NAME depends on organization and what variables matter

                    out_name = 'Histories/stats'+'-'+str(batch_size)+'-'+str(learning_rate)+'-'+str(decay_rate)+'-'+str(latent_dim)+'.pickle'
                    Conv_VAE3D(n_epochs,
                               batch_size,
                               learning_rate,
                               decay_rate,
                               latent_dim,
                               out_name)
                    
