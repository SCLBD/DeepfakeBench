import random

import torch


def augment_domains(self, groups_feature_maps):
    # Helper Functions
    def hard_example_interpolation(z_i, hard_example, lambda_1):
        return z_i + lambda_1 * (hard_example - z_i)

    def hard_example_extrapolation(z_i, mean_latent, lambda_2):
        return z_i + lambda_2 * (z_i - mean_latent)

    def add_gaussian_noise(z_i, sigma, lambda_3):
        epsilon = torch.randn_like(z_i) * sigma
        return z_i + lambda_3 * epsilon

    def difference_transform(z_i, z_j, z_k, lambda_4):
        return z_i + lambda_4 * (z_j - z_k)

    def distance(z_i, z_j):
        return torch.norm(z_i - z_j)

    domain_number = len(groups_feature_maps[0])

    # Calculate the mean latent vector for each domain across all groups
    domain_means = []
    for domain_idx in range(domain_number):
        all_samples_in_domain = torch.cat([group[domain_idx] for group in groups_feature_maps], dim=0)
        domain_mean = torch.mean(all_samples_in_domain, dim=0)
        domain_means.append(domain_mean)

    # Identify the hard example for each domain across all groups
    hard_examples = []
    for domain_idx in range(domain_number):
        all_samples_in_domain = torch.cat([group[domain_idx] for group in groups_feature_maps], dim=0)
        distances = torch.tensor([distance(z, domain_means[domain_idx]) for z in all_samples_in_domain])
        hard_example = all_samples_in_domain[torch.argmax(distances)]
        hard_examples.append(hard_example)

    augmented_groups = []

    for group_feature_maps in groups_feature_maps:
        augmented_domains = []

        for domain_idx, domain_feature_maps in enumerate(group_feature_maps):
            # Choose a random augmentation
            augmentations = [
                lambda z: hard_example_interpolation(z, hard_examples[domain_idx], random.random()),
                lambda z: hard_example_extrapolation(z, domain_means[domain_idx], random.random()),
                lambda z: add_gaussian_noise(z, random.random(), random.random()),
                lambda z: difference_transform(z, domain_feature_maps[0], domain_feature_maps[1], random.random())
            ]
            chosen_aug = random.choice(augmentations)
            augmented = torch.stack([chosen_aug(z) for z in domain_feature_maps])
            augmented_domains.append(augmented)

        augmented_domains = torch.stack(augmented_domains)
        augmented_groups.append(augmented_domains)

    return torch.stack(augmented_groups)


def mixup_in_latent_space(self, data):
    # data shape: [batchsize, num_domains, 3, 256, 256]
    bs, num_domains, _, _, _ = data.shape

    # Initialize an empty tensor for mixed data
    mixed_data = torch.zeros_like(data)

    # For each sample in the batch
    for i in range(bs):
        # Step 1: Generate a shuffled index list for the domains
        shuffled_idxs = torch.randperm(num_domains)

        # Step 2: Choose random alpha between 0.5 and 2, then sample lambda from beta distribution
        alpha = torch.rand(1) * 1.5 + 0.5  # random alpha between 0.5 and 2
        lambda_ = torch.distributions.beta.Beta(alpha, alpha).sample().to(data.device)

        # Step 3: Perform mixup using the shuffled indices
        mixed_data[i] = lambda_ * data[i] + (1 - lambda_) * data[i, shuffled_idxs]

    return mixed_data