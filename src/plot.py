import os
import argparse
import numpy as np
import seaborn
import pandas
import matplotlib.pyplot as plt


def test_plot(run_folder):
	pass


def make_plots(run_folder):
	
	seaborn.set(rc={'figure.figsize':(11.7,8.27)})

	csv = pandas.read_csv(os.path.join(run_folder, "losses.csv"))
	line1 = seaborn.lineplot(x="transitions trained", y="actor loss", data=csv, linewidth=0.0)
	ax = plt.twiny()
	line2 = seaborn.lineplot(x="transitions gathered", y="actor loss", data=csv, linewidth=0.5)
	line2.fill_between(csv["transitions gathered"], y1=csv["actor loss"] - csv["actor loss std"], y2=csv["actor loss"] + csv["actor loss std"], alpha=0.5)
	plt.savefig(os.path.join(run_folder, "actor_loss.svg"))

	plt.clf()
	line1 = seaborn.lineplot(x="transitions trained", y="critic loss", data=csv, linewidth=0.0)
	ax = plt.twiny()
	line2 = seaborn.lineplot(x="transitions gathered", y="critic loss", data=csv, linewidth=0.5)
	line2.fill_between(csv["transitions gathered"], y1=csv["critic loss"] - csv["critic loss std"], y2=csv["critic loss"] + csv["critic loss std"], alpha=0.5)
	plt.savefig(os.path.join(run_folder, "critic_loss.svg"))

	plt.clf()
	line1 = seaborn.lineplot(x="transitions trained", y="average Q", data=csv, linewidth=0.0)
	ax = plt.twiny()
	line2 = seaborn.lineplot(x="transitions gathered", y="average Q", data=csv, linewidth=0.75)
	line2.fill_between(csv["transitions gathered"], y1=csv["average Q"] - csv["average Q std"], y2=csv["average Q"] + csv["average Q std"], alpha=0.5)
	plt.savefig(os.path.join(run_folder, "q_vals.svg"))
	
	plt.clf()
	csv = pandas.read_csv(os.path.join(run_folder, "tests.csv"))
	line1 = seaborn.lineplot(x="transitions trained", y="average episode return", data=csv, linewidth=0.0)
	ax = plt.twiny()
	line2 = seaborn.lineplot(x="transitions gathered", y="average episode return", data=csv, linewidth=0.75)
	line2.fill_between(csv["transitions gathered"], y1=csv["average episode return"] - csv["avg ep ret std"], y2=csv["average episode return"] + csv["avg ep ret std"], alpha=0.5)
	plt.savefig(os.path.join(run_folder, "tests.svg"))


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
	parser.add_argument('--run_folder', type=str, help='Which run to plot (foldername).', required=True)

	args = parser.parse_args()
	
	make_plots(args.run_folder)
