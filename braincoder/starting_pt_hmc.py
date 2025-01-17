import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

# Set random seed for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

tfp = tfp.experimental

tfd = tfp.distributions

# Generate synthetic data
true_weights = [2.0, -1.0]  # True slope and intercept
num_samples = 100
x_data = np.random.uniform(-1, 1, size=num_samples)
y_data = true_weights[0] * x_data + true_weights[1] + np.random.normal(0, 0.1, size=num_samples)

# Define the model
def linear_model(params, x):
    slope, intercept = params
    return slope * x + intercept

# Define the log-likelihood
@tf.function
def log_likelihood(params):
    slope, intercept = params
    predicted_y = linear_model([slope, intercept], x_data)
    return tf.reduce_sum(tfd.Normal(loc=predicted_y, scale=0.1).log_prob(y_data))

# Define the log-prior
@tf.function
def log_prior(params):
    slope, intercept = params
    return tfd.Normal(0, 10).log_prob(slope) + tfd.Normal(0, 10).log_prob(intercept)

# Define the unnormalized log-posterior
@tf.function
def unnormalized_log_posterior(params):
    return log_likelihood(params) + log_prior(params)

# Initial values for the parameters
initial_params = [0.0, 0.0]

# Set up the Hamiltonian Monte Carlo algorithm
step_size = 0.1
num_results = 500
num_burnin_steps = 200

kernel = tfp.mcmc.HamiltonianMonteCarlo(
    target_log_prob_fn=unnormalized_log_posterior,
    step_size=step_size,
    num_leapfrog_steps=3
)

# Use the TransformedTransitionKernel to automatically handle unconstraining transformations
adaptive_kernel = tfp.mcmc.DualAveragingStepSizeAdaptation(
    inner_kernel=kernel,
    num_adaptation_steps=int(0.8 * num_burnin_steps),
    target_accept_prob=0.75
)

@tf.function
def run_chain():
    return tfp.mcmc.sample_chain(
        num_results=num_results,
        current_state=initial_params,
        kernel=adaptive_kernel,
        num_burnin_steps=num_burnin_steps,
        trace_fn=lambda current_state, kernel_results: kernel_results.is_accepted
    )

# Run the MCMC chain
samples, is_accepted = run_chain()

# Extract samples
slope_samples, intercept_samples = samples

# Print results
print("Acceptance rate:", np.mean(is_accepted))
print("Mean slope:", np.mean(slope_samples))
print("Mean intercept:", np.mean(intercept_samples))

# Visualize the posterior distributions
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(slope_samples, bins=30, color='blue', alpha=0.7, label='Slope')
plt.axvline(true_weights[0], color='red', linestyle='--', label='True slope')
plt.legend()
plt.title("Posterior of Slope")

plt.subplot(1, 2, 2)
plt.hist(intercept_samples, bins=30, color='green', alpha=0.7, label='Intercept')
plt.axvline(true_weights[1], color='red', linestyle='--', label='True intercept')
plt.legend()
plt.title("Posterior of Intercept")

plt.tight_layout()
plt.show()
