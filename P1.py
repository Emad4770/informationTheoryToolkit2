import numpy as np
from sklearn.neighbors import KernelDensity

vector_length = 100

def pmf_univariate(samples):
    n = len(samples)
    samples_list, pmf_vector = np.unique(samples, return_counts=True)
    return samples_list, pmf_vector/n

def entropy(pmf_vector):
    ent = -np.sum(pmf_vector * np.log2(pmf_vector)) #calculating entropy
    return ent


pmf_vector = np.array([0.2, 0.3, 0.1, 0.4])
original_entropy = entropy(pmf_vector)

# np.random.seed(42)  #for reproducibility
samples = np.random.choice(np.arange(len(pmf_vector)), size=vector_length, p=pmf_vector)
                              #[0, 1, 2, 3]                       #Probabilities

_ , estimated_pmf_vector = pmf_univariate(samples)
estimated_entropy = entropy(estimated_pmf_vector)

entropy_difference = np.abs(original_entropy - estimated_entropy)

print("Original Entropy:", original_entropy)
print("Estimated Entropy:", estimated_entropy)
print("Difference in Entropy:", entropy_difference)


###

def differential_entropy(pdf):
    pdf = np.where(pdf == 0, 1e-12, pdf)  #replace 0 values with a small positive value
    entropy = -np.trapz(pdf * np.log2(pdf), dx=1e-6) #integration
    return entropy

####

mean = 0
std = 1
samples_continues = np.random.normal(mean, std, size=vector_length)
optimal_bandwidth = 1.06 * std * np.power(len(samples_continues), -1/5)
bandwidth = 0.5
kernel = 'tophat'


# bandwidths =[0.1,0.4,1.0, optimal_bandwidth]
# kernels = ['gaussian', 'tophat', 'epanechnikov','exponential','linear','cosine']
# vector_lengths = [100, 1000, 10000]




def gaussian_pdf(x, mean, std):
    exponent = -((x - mean) ** 2) / (2 * std ** 2)
    normalization = 1 / (std * np.sqrt(2 * np.pi))
    pdf = normalization * np.exp(exponent) #gausin pdf formula
    return pdf


def kd_estimation(samples_continues, kernel, bandwidth):
    #Estimate the pdf using Kernel Density Estimation
    kde = KernelDensity(kernel= kernel,  bandwidth=bandwidth).fit(samples_continues.reshape(-1, 1))
    x_values = np.linspace(-10, 10, 1000)
    log_density_estimate = kde.score_samples(x_values.reshape(-1, 1))
    estimated_pdf = np.exp(log_density_estimate)
    #Normalize the pdf so the sum is 1
    estimated_pdf = estimated_pdf / np.sum(estimated_pdf)

    return estimated_pdf

# print('ga pdf= ', gaussian_pdf(samples,mean,std))
# print('entropy gaussian= ', differential_entropy(0.12))
# print('Pdf est= ', estimated_pdf)
gaussian_diff_entropy = differential_entropy(gaussian_pdf(samples_continues,mean,std))
estimated_diff_entropy = differential_entropy(kd_estimation(samples_continues,kernel,bandwidth))
diff_ent_difference = np.abs(gaussian_diff_entropy - estimated_diff_entropy)

print('Gaussian pdf Entropy: ', gaussian_diff_entropy)
print('Estimated pdf Entropy: ',   estimated_diff_entropy)
print('Difference in Differential Entropy: ',"%.10f" % diff_ent_difference)

