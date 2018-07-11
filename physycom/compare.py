import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

prefix="count/model_perf/"
photo_dir=["SHT_photo", "VEN_photo", "STUD_photo", "UCF_photo"]

results = {}
normres = {}
for pdir in photo_dir:
  dataset=pdir.split("_")[0]
  results[dataset] = {}
  normres[dataset] = {}
  for file in os.listdir(prefix + pdir):
    if file.endswith("_compare.csv"):
      #print("FILE ",file)
      data = pd.read_csv(prefix + pdir + "/" + file, sep=';')
      for _, row in data.iterrows():
        if row['model'] in results[dataset].keys():
          results[dataset][row['model']] += [row['err']]
          normres[dataset][row['model']] += [row['err']/row['gt']]
        else:
          results[dataset][row['model']] = [row['err']]
          normres[dataset][row['model']] = [row['err']/row['gt']]

# collect results
rmse = {}
mae = {}
for key, vec in results.items():
  rmse[key] = {}
  mae[key] = {}
  for label, vals in vec.items():
    rmse[key][label] = np.sqrt(np.mean(np.asarray(vals)**2))
    mae[key][label] = np.mean(np.abs(np.asarray(vals)))

# collect normalized results
nrmse = {}
nmae = {}
for key, vec in normres.items():
  nrmse[key] = {}
  nmae[key] = {}
  for label, vals in vec.items():
    nrmse[key][label] = np.sqrt(np.mean(np.asarray(vals)**2))
    nmae[key][label] = np.mean(np.abs(np.asarray(vals)))

[ plt.plot(vals.values(), 'o-', label=key) for key, vals in rmse.items() ]
plt.xticks(np.arange(0,12), rmse['SHT'].keys(), rotation='45')
plt.gca().set_xticklabels(rmse['SHT'].keys())
plt.xlabel('Pre-trained model')
plt.ylabel('RMSE')
plt.legend()
plt.title('Root Mean Squared Error for various dataset')
plt.grid(linestyle=':')
plt.savefig("compare_RMSE.png")
plt.gcf().clear()

[ plt.plot(vals.values(), 'o-', label=key) for key, vals in mae.items() ]
plt.xticks(np.arange(0,12), mae['SHT'].keys(), rotation='45')
plt.gca().set_xticklabels(mae['SHT'].keys())
plt.xlabel('Pre-trained model')
plt.ylabel('MAE')
plt.legend()
plt.title('Mean Absolute Error for various dataset')
plt.grid(linestyle=':')
plt.savefig("compare_MAE.png")
plt.gcf().clear()

[ plt.plot(vals.values(), 'o-', label=key) for key, vals in nrmse.items() ]
plt.xticks(np.arange(0,12), nrmse['SHT'].keys(), rotation='45')
plt.gca().set_xticklabels(nrmse['SHT'].keys())
plt.xlabel('Pre-trained model')
plt.ylabel('Normalized RMSE')
plt.legend()
plt.title('Normalized Root Mean Squared Error for various dataset')
plt.grid(linestyle=':')
plt.savefig("compare_NRMSE.png")
plt.gcf().clear()

[ plt.plot(vals.values(), 'o-', label=key) for key, vals in nmae.items() ]
plt.xticks(np.arange(0,12), nmae['SHT'].keys(), rotation='45')
plt.gca().set_xticklabels(nmae['SHT'].keys())
plt.xlabel('Pre-trained model')
plt.ylabel('Normalized MAE')
plt.legend()
plt.title('Normalized Mean Absolute Error for various dataset')
plt.grid(linestyle=':')
plt.savefig("compare_NMAE.png")
plt.gcf().clear()
