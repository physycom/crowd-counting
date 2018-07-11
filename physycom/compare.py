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

def save_results(res, y_label, title, out_name):
  [ plt.plot(vals.values(), 'o-', label=key) for key, vals in res.items() ]
  labels = res[list(res.keys())[0]].keys()
  tics = np.arange(0,len(labels)+1)
  plt.xticks(tics, labels, rotation='45')
  plt.gca().set_xticklabels(res['SHT'].keys())
  plt.xlabel('Pre-trained model')
  plt.ylabel(y_label)
  plt.title(title)
  plt.legend()
  plt.grid(linestyle=':')
  plt.tight_layout()
  plt.savefig(out_name, dpi=200)
  plt.gcf().clear()

save_results( rmse, 'RMSE',            'Root Mean Squared Error for various dataset',  'compare_RMSE.png')
save_results(  mae,  'MAE',                'Mean Absolute Error for various dataset',   'compare_MAE.png')
save_results(nrmse, 'RMSE', 'Normalized Root Mean Squared Error for various dataset', 'compare_NRMSE.png')
save_results( nmae, 'RMSE',     'Normalized Mean Absolute Error for various dataset',  'compare_NMAE.png')
