# Function to wrap around ConvVAE_func.py
# Scans parameter ranges (grid search approach)

from ConvVAE_func import Conv_VAE


n_epochs_range = [100]
batch_size_range = [32, 64, 128, 256]
learning_rate_range = [0.001]
decay_rate_range = [0.0]
latent_dim_range = [2]


for n_epochs in n_epochs_range:
    for batch_size in batch_size_range:
        for learning_rate in learning_rate_range:
            for decay_rate in decay_rate_range:
                for latent_dim in latent_dim_range:
                    # IMPORTANT:
                    # OUTPUT NAME depends on organization and what variables matter

                    out_name = 'Histories/stats'+'-'+str(batch_size)+'-'+str(learning_rate)+'-'+str(decay_rate)+'.pickle'
                    Conv_VAE(n_epochs,
                             batch_size,
                             learning_rate,
                             decay_rate,
                             latent_dim,
                             out_name)
                    
