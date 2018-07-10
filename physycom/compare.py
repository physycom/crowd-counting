import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

prefix="count/model_perf/"
photo_dir=["SHT_photo", "CAM_photo", "OUR_photo"]

results = {}
for pdir in photo_dir:
  dataset=pdir.split("_")[0]
  results[dataset] = {}
  for file in os.listdir(prefix + pdir):
    if file.endswith("_compare.csv"):
      #print("FILE ",file)
      data = pd.read_csv(prefix + pdir + "/" + file, sep=';')
      for _, row in data[['model','err']].iterrows():
        if row['model'] in results[dataset].keys():
          results[dataset][row['model']] += [row['err']]
        else:
          results[dataset][row['model']] = [row['err']]

rss = {}
for key, vec in results.items():
  rss[key] = {}
  for label, vals in vec.items():
    rss[key][label] = np.sqrt(np.mean(np.asarray(vals)**2))

[ plt.plot(vals.values(), 'o-', label=key) for key, vals in rss.items() ]
plt.xticks(np.arange(0,12), rss['SHT'].keys(), rotation='45')
plt.gca().set_xticklabels(rss['SHT'].keys())
plt.legend()
plt.show()
plt.grid(linestyle=':')
