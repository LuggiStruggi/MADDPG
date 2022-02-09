import os
import argparse
import numpy as np
import seaborn
import pandas
import matplotlib.pyplot as plt


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
	parser.add_argument('--plot_folder', type=str, help='Which saved weights to test (foldername).', required=True)

	args = parser.parse_args()

	csv = pandas.read_csv(os.path.join(args.plot_folder, "losses.csv"))
	line1 = seaborn.lineplot(x="transitions made", y="actor loss", data=csv, linewidth=0.5)
	line1.fill_between(csv["transitions made"], y1=csv["actor loss"] - csv["actor_loss_std"], y2=csv["actor loss"] + csv["actor_loss_std"], alpha=0.5)
	line2 = seaborn.lineplot(x="transitions made", y="critic loss", data=csv, linewidth=0.5)
	line2.fill_between(csv["transitions made"], y1=csv["critic loss"] - csv["critic_loss_std"], y2=csv["critic loss"] + csv["critic_loss_std"], alpha=0.5)
	plt.savefig(os.path.join(args.plot_folder, "losses.svg"))
	
	plt.clf()
	csv = pandas.read_csv(os.path.join(args.plot_folder, "tests.csv"))
	line1 = seaborn.lineplot(x="transitions made", y="average episode return", data=csv, linewidth=0.5)
	line1.fill_between(csv["transitions made"], y1=csv["average episode return"] - csv["avg_ep_ret_std"], y2=csv["average episode return"] + csv["avg_ep_ret_std"], alpha=0.5)
	plt.savefig(os.path.join(args.plot_folder, "tests.svg"))
