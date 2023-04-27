import matplotlib.pyplot as plt
import numpy as np
from single_digit import KL


def compare(unmarked_file, watermarked_file, out_file):
    
    unmarked_dists = np.load(unmarked_file)
    watermarked_dists = np.load(watermarked_file)

    kl_data = []
    for u_d in unmarked_dists:
        for w_d in watermarked_dists:
            kl = KL(u_d, w_d)
            kl_data.append(kl_data)

    fig, ax = plt.subplots()
    ax.violinplot(kl_data, showmeans=False, showmedians=True)
    ax.set_title('Distribution of Pairwise KLs Between Unmarked and Watermarked Models')
    ax.set_ylabel('KL Divergence')
    ax.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)

    plt.savefig(out_file)

    return KL(dist_unmarked, dist_watermarked)


if __name__ == "__main__":
    compare('td3_marked', 'td3_unmarked', 'td3_compare.png')
