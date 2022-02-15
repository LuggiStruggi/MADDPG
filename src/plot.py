import os
import argparse
import numpy as np
import seaborn
import pandas
import matplotlib.pyplot as plt


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser("Parser to Initiate Agent Training")
	parser.add_argument('--run_folder', type=str, help='Which run to plot (foldername).', required=True)

	args = parser.parse_args()
	
	seaborn.set(rc={'figure.figsize':(11.7,8.27)})

	csv = pandas.read_csv(os.path.join(args.run_folder, "losses.csv"))
	line1 = seaborn.lineplot(x="transitions made", y="actor loss", data=csv, linewidth=0.5)
	line1.fill_between(csv["transitions made"], y1=csv["actor loss"] - csv["actor loss std"], y2=csv["actor loss"] + csv["actor loss std"], alpha=0.5)	
	line2 = seaborn.lineplot(x="transitions made", y="critic loss", data=csv, linewidth=0.5)
	line2.fill_between(csv["transitions made"], y1=csv["critic loss"] - csv["critic loss std"], y2=csv["critic loss"] + csv["critic loss std"], alpha=0.5)
	plt.savefig(os.path.join(args.run_folder, "losses.svg"))

	plt.clf()
	csv = pandas.read_csv(os.path.join(args.run_folder, "losses.csv"))
	line1 = seaborn.lineplot(x="transitions made", y="average Q", data=csv, linewidth=0.75)
	line1.fill_between(csv["transitions made"], y1=csv["average Q"] - csv["average Q std"], y2=csv["average Q"] + csv["average Q std"], alpha=0.5)
	plt.savefig(os.path.join(args.run_folder, "q_vals.svg"))
	
	plt.clf()
	csv = pandas.read_csv(os.path.join(args.run_folder, "tests.csv"))
	line1 = seaborn.lineplot(x="transitions made", y="average episode return", data=csv, linewidth=0.75)
	line1.fill_between(csv["transitions made"], y1=csv["average episode return"] - csv["avg ep ret std"], y2=csv["average episode return"] + csv["avg ep ret std"], alpha=0.5)
	plt.savefig(os.path.join(args.run_folder, "tests.svg"))
