from transformers import SegformerImageProcessor, AutoModelForSemanticSegmentation
from PIL import Image
import requests
import matplotlib.pyplot as plt
import torch.nn as nn

import torch
from matplotlib.patches import Patch

processor = SegformerImageProcessor.from_pretrained('sayeed99/segformer_b3_clothes')
model = AutoModelForSemanticSegmentation.from_pretrained('sayeed99/segformer_b3_clothes')

url = 'https://plus.unsplash.com/premium_photo-1673210886161-bfcc40f54d1f?ixlib=rb-4.0.3&ixid=MnwxMjA3fDB8MHxzZWFyY2h8MXx8cGVyc29uJTIwc3RhbmRpbmd8ZW58MHx8MHx8&w=1000&q=80'

img = Image.open('top_influenceurs_2024/IMG/image_0.png').convert('RGB')
#img = Image.open(requests.get(url, stream=True).raw)
inputs = processor(images=img, return_tensors='pt')

print(tuple(inputs['pixel_values'].shape))
print(round(inputs['pixel_values'].min().item(), 2), 
      '/', 
      round(inputs["pixel_values"].max().item(), 2))

outputs = model(**inputs)
logits = outputs.logits.cpu()

print(tuple(logits.shape))

upsampled_logits = nn.functional.interpolate(
    logits,
    size=img.size[::-1],
    mode='bilinear',
    align_corners=False,
)

print(tuple(upsampled_logits.shape))

pred_seg = upsampled_logits.argmax(dim=1)[0]

print(tuple(pred_seg.shape))

y, x = img.height // 2, img.width // 2
probas = upsampled_logits[0, :, y, x].softmax(dim=0)
for p, i in zip(*probas.topk(3)):
    print(f'{model.config.id2label[i.item()]:15s} {p.item() * 100:5.1f} %')

ids, counts = torch.unique(pred_seg, return_counts=True)
for i, c in sorted(zip(ids.tolist(), counts.tolist()), key=lambda t: -t[1]):
    print(f'{model.config.id2label[i]:15s} {100 * c / pred_seg.numel():5.1f} %')

plt.imshow(pred_seg)
plt.imshow(pred_seg)
plt.axis("off")
plt.show()

mean = torch.tensor(processor.image_mean).view(3, 1, 1)
std = torch.tensor(processor.image_std).view(3, 1, 1)

vue_par_le_modele = (inputs["pixel_values"][0] * std + mean).clamp(0, 1).permute(1, 2, 0)
carte_grille = logits.argmax(dim=1)[0]

norm = plt.Normalize(0, 17)
cmap = plt.cm.tab20

fig, axes = plt.subplots(1, 4, figsize=(18, 7))
axes[0].imshow(img)
axes[0].set_title(f'1. Photo originale\n{img.width} x {img.height}')
axes[1].imshow(vue_par_le_modele)
axes[1].set_title('2. Ce que reçoit le modèle\n512 x 512 (déformée)')
axes[2].imshow(carte_grille, cmap=cmap, norm=norm, interpolation='nearest')
axes[2].set_title('3. Prédiction sur la grille\n128 x 128 (avant agrandissement)')
axes[3].imshow(pred_seg, cmap=cmap, norm=norm)
axes[3].set_title(f'4. Prédiction finale\n{img.width} x {img.height}')
for ax in axes:
    ax.axis("off")

legende = [Patch(facecolor=cmap(norm(i)), label=model.config.id2label[i])
           for i in ids.tolist()]
fig.legend(handles=legende, loc='lower center', ncol=len(legende), frameon=False)
plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.show()
