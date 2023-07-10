import pickle

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'text.usetex': True,
})
font = {'size': 20}
plt.rcParams['figure.dpi'] = 450
plt.rc('font', family='serif', serif='Times', size=20)
plt.rc('text', usetex=True)
plt.rc('axes', axisbelow=True)

# matplotlib.rc('font', **font)

COLOR = '#dc3f76'
BLUE = '#001482'

with open('/home/william/Datasets/SemanticKITTI/mask_area_no_completion.pkl', 'rb') as f:
    area_no_completion_gt = pickle.load(f)

with open('/home/william/Datasets/SemanticKITTI/mask_area_completion.pkl', 'rb') as f:
    area_completion_gt = pickle.load(f)

with open('/home/william/Datasets/SemanticKITTI/pred_area.pkl', 'rb') as f:
    area_pred = pickle.load(f)

area_no_completion_gt = area_no_completion_gt[8]
area_completion_gt = area_completion_gt[8]

# Pred percent
percent_area_pred = []
for inst, masks in area_pred.items():
    tgt, pred = masks['tgt'].cpu(), masks['pred'].cpu()
    if tgt > 0:
        percent_area_pred.append(pred / tgt)

# plt.title('')
w, h = plt.figaspect(0.9)
plt.figure(figsize=(w, h))
plt.xlabel(r'$A_{pred} / A_{complete}$')
plt.ylabel('\# Instances')
plt.xlim((0, 3))
plt.gca().yaxis.grid(True, color='dimgray', linestyle='--')
plt.hist(percent_area_pred, bins=25, color=COLOR)
mean_percent_pred = np.mean(percent_area_pred)
print(mean_percent_pred)
plt.axvline(x=mean_percent_pred, ymin=0, ymax=70, color=BLUE)
plt.text(1.36, 68, '$\mu=1.29$', rotation=0, fontsize=20, color=BLUE)
plt.subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig('/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/area_pred.png')
plt.show()


# single scan percent
percent_single_area = []
area_at_zero = []
for inst in area_completion_gt.keys():
    single_area = area_no_completion_gt[inst]
    full_area = area_completion_gt[inst]
    if single_area < 10:
        area_at_zero.append(full_area)
    percent_single_area.append(single_area / full_area)
percent_single_area = np.array(percent_single_area)
# plt.title('Single instance mask / full instance mask')
mean_percent_area = np.mean(percent_single_area)
print(f'{mean_percent_area=}')
w, h = plt.figaspect(0.9)
plt.figure(figsize=(w, h))
plt.xlabel(r'$A_{single} / A_{complete}$')
plt.ylabel('\# Instances')
plt.xlim((0, 1))
plt.hist(percent_single_area, bins=20, color=COLOR)
plt.gca().yaxis.grid(True, color='dimgray', linestyle='--')
mean_no_outlier = np.mean(percent_single_area[percent_single_area > 0.1])
print(f'{mean_no_outlier=}')
# plt.axvline(x=mean_no_outlier, ymin=0, ymax=70, ls='--', color=BLUE)
# plt.text(0.70, 48, '$\mu=0.61$', rotation=0, fontsize=20, color=BLUE)
plt.axvline(x=mean_percent_area, ymin=0, ymax=70, color=BLUE)
plt.text(0.27, 43.5, '$\mu=0.55$', rotation=0, fontsize=20, color=BLUE)
plt.subplots_adjust(bottom=0.15)
plt.tight_layout()
plt.savefig('/home/william/Documents/Writing/publication_IROS2023_WilliamGuimont-Martin/figs/area_single.png')
plt.show()

exit(0)

plt.hist(area_at_zero)
plt.show()

print(f'{np.mean_percent_pred(percent_single_area)=}')
print(f'{np.std(percent_single_area)=}')

print(f'{len(area_no_completion_gt)=}')
print(f'{len(area_completion_gt)=}')
print(f'{area_pred}')

no_completion_gt_values = list(area_no_completion_gt.values())
completion_gt_values = list(area_completion_gt.values())

plt.hist(no_completion_gt_values, label='Area of mask from a single scan', ls='dashed', lw=3, fc=(0, 0, 1, 0.5))
plt.hist(completion_gt_values, label='Area of complete mask', ls='dotted', lw=3, fc=(1, 0, 0, 0.5))
plt.show()

print(f'{np.mean_percent_pred(no_completion_gt_values)=}')
print(f'{np.std(no_completion_gt_values)=}')
print(f'{np.mean_percent_pred(completion_gt_values)=}')
print(f'{np.std(completion_gt_values)=}')
