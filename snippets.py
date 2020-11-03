import numpy as np

def discrete_sampling(distribution, num_samples,
                   min, max):
  disc = (max - min)/distribution.shape[0]
  domain = np.arange(min, max, disc)
  cdf = np.cumsum(distribution)
  uniforms = np.random.rand(num_samples)
  output_samples = []
  for u in uniforms:
    for i in range(len(cdf)):
      if u <= cdf[i]:
        output_samples.append(domain[i])
        break
  return output_samples


distribution = [0, 0, 0, 0, 0, 0.2, 0.2, 0.2, 0.2, 0.2]
samples = discrete_sampling(np.array(distribution), 20, 0, 10)
print(samples)
