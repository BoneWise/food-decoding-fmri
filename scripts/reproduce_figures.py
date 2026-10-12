#!/usr/bin/env python3
"""Draw portable table-driven figures; archived cortical surfaces are separate assets."""
from __future__ import annotations
import argparse
import json
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from reproduce_results import ROOT, REFERENCE, DIVISIONS, reproduce

COLORS = ['#26768D', '#C66B39']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':12,
    'axes.labelsize':10,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.facecolor':'white'})


def save(fig, output, name):
    fig.savefig(output / f'{name}.png', dpi=220, bbox_inches='tight')
    fig.savefig(output / f'{name}.pdf', bbox_inches='tight',
                metadata={'Creator':'food-subjective-value-fmri table reproduction', 'CreationDate':None, 'ModDate':None})
    plt.close(fig)


def errorbar(ax, mean, low, high, position, color, label=None, horizontal=False):
    error = np.array([[a-b for a,b in zip(mean, low)], [a-b for a,b in zip(high, mean)]])
    if horizontal:
        ax.errorbar(mean, position, xerr=error, fmt='o', color=color, capsize=3, ms=6, label=label)
    else:
        ax.errorbar(position, mean, yerr=error, fmt='o', color=color, capsize=3, ms=6, label=label)


def performance(tables, output):
    fig, ax = plt.subplots(figsize=(5.8,4),layout='constrained')
    rows = tables['decoding_performance.csv']
    for i, model in enumerate(['CNN','SVM']):
        selected=[r for r in rows if r['model']==model]
        x=np.arange(3)+(i-.5)*.18
        mean=[r['participant_mean_ba']*100 for r in selected]
        errorbar(ax,mean,[r['ci_low_frozen']*100 for r in selected],
                 [r['ci_high_frozen']*100 for r in selected],x,COLORS[i],model)
        for p,m in zip(x,mean):ax.annotate(f'{m:.2f}',(p,m),xytext=(5,5),textcoords='offset points',fontsize=8)
    ax.axhline(50,color='#8A9298',ls='--',lw=1)
    ax.set(xticks=range(3),xticklabels=['k = 1','k = 3','k = 6'],ylim=(48,72),
        ylabel='Participant balanced accuracy (%)',xlabel='Number of beta maps per input',
        title='Cross-participant decoding of\nsubjective food value')
    ax.legend(frameon=False,loc='upper left');ax.grid(axis='y',alpha=.15)
    save(fig,output,'decoding_performance')


def regional(tables,output):
    fig,ax=plt.subplots(figsize=(6.3,3.8),layout='constrained')
    rows=tables['layercam_macroregions.csv']
    for i,rating in enumerate([1,4]):
        data={r['division']:r for r in rows if r['rating']==rating}
        vals=[data[d]['mean_layercam_whole_brain_1'] for d in DIVISIONS]
        ax.barh(np.arange(5)+(i-.5)*.34,vals,height=.31,color=COLORS[i],
                label='Low subjective food value' if rating==1 else 'High subjective food value')
    ax.axvline(1,color='#77838B',ls='--',lw=1)
    ax.set(yticks=range(5),yticklabels=[d if d!='Subcortical' else 'Insular/subcortical' for d in DIVISIONS],xlim=(0,1.7),
        xlabel='Mean Layer-CAM value (whole-brain mean = 1)',title='Regional distribution of CNN spatial evidence')
    ax.invert_yaxis();ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.20),fontsize=8)
    save(fig,output,'layercam_macroregions')
    selected=[('Right Parahippocampal Gyrus, posterior division','R posterior parahippocampal gyrus'),
        ('Right Right Hippocampus','R hippocampus'),('Right Precuneous Cortex','R precuneus'),
        ('Left Precuneous Cortex','L precuneus'),('Right Cingulate Gyrus, posterior division','R posterior cingulate'),
        ('Left Cingulate Gyrus, posterior division','L posterior cingulate'),
        ('Right Insular Cortex','R insula'),('Left Left Accumbens','L accumbens'),
        ('Right Right Accumbens','R accumbens'),('Right Right Thalamus','R thalamus')]
    all_rois={(r['rating'],r['roi_name']):r for r in tables['layercam_all_rois.csv']}
    fig,ax=plt.subplots(figsize=(7.5,5.1),layout='constrained')
    for i,rating in enumerate([1,4]):
        vals=[all_rois[(rating,name)]['mean_layercam_whole_brain_1'] for name,_ in selected]
        ax.plot(vals,np.arange(len(selected))+(i-.5)*.2,'o',color=COLORS[i],ms=6,
                label='Low subjective food value' if rating==1 else 'High subjective food value')
        for j,value in enumerate(vals):
            ax.annotate(f'{value:.3f}',(value,j+(i-.5)*.2),xytext=(6,0),
                        textcoords='offset points',va='center',fontsize=7,color=COLORS[i])
    ax.axvline(1,color='#77838B',ls='--',lw=1)
    ax.set(yticks=range(len(selected)),yticklabels=[label for _,label in selected],xlim=(.95,1.76),
        xlabel='Mean Layer-CAM value (whole-brain mean = 1)',title='Selected regional values in the k = 6 CNN')
    ax.invert_yaxis();ax.grid(axis='y',alpha=.12)
    ax.legend(frameon=False,loc='upper center',bbox_to_anchor=(.5,-.16),ncol=2,fontsize=8)
    save(fig,output,'layercam_selected_rois')
    for rating,color in zip([1,4],COLORS):
        fig,axes=plt.subplots(5,1,figsize=(9,13),layout='constrained')
        for ax,division in zip(axes,DIVISIONS):
            values=[r for r in tables['layercam_top5_rois.csv'] if r['rating']==rating and r['division']==division]
            names=[r['roi_name'].replace('Left Left','Left').replace('Right Right','Right').replace('Precuneous','Precuneus') for r in values]
            names=['\n'.join(textwrap.wrap(n,43)) for n in names]
            ax.barh(range(5),[r['mean_layercam_whole_brain_1'] for r in values],color=color,height=.62)
            ax.set(yticks=range(5),yticklabels=names,title=division if division!='Subcortical' else 'Insular/subcortical',xlim=(0,2.6))
            ax.invert_yaxis();ax.axvline(1,color='#77838B',ls='--',lw=1);ax.tick_params(axis='y',labelsize=8)
        axes[-1].set_xlabel('Mean Layer-CAM value (whole-brain mean = 1)')
        fig.suptitle(('Low' if rating==1 else 'High')+' subjective food value:\nfive highest-valued ROIs per macroregion',fontsize=14)
        save(fig,output,f'layercam_top5_rating{rating}')


def supplement(tables,output):
    rows=tables['input_replacement_summary.csv']
    labels={'full':'Full input','cam_top20':'CAM top 20%','random_top20_mean':'Random top 20%',
            'value':'Value ROIs','visual':'Visual ROIs','union':'Value + visual ROIs'}
    fig,axes=plt.subplots(1,2,figsize=(11,4.5),layout='constrained',gridspec_kw={'width_ratios':[1,1.15]})
    ax=axes[0]
    errorbar(ax,[r['mean_ba']*100 for r in rows],[r['ci_low_frozen']*100 for r in rows],
             [r['ci_high_frozen']*100 for r in rows],range(len(rows)),COLORS[0],horizontal=True)
    ax.set(yticks=range(len(rows)),yticklabels=[labels[r['condition']] for r in rows],
        xlabel='Participant balanced accuracy (%)',title='A  Input replacement')
    ax.invert_yaxis();ax.axvline(50,color='#77838B',ls='--',lw=1)
    rows=tables['input_replacement_inference_frozen.csv'];ax=axes[1]
    names=['Random − CAM','Full − value','Full − visual','Full − union']
    errorbar(ax,[float(r['mean_effect'])*100 for r in rows],[float(r['ci_low'])*100 for r in rows],
             [float(r['ci_high'])*100 for r in rows],range(4),COLORS[1],horizontal=True)
    ax.set(yticks=range(4),yticklabels=names,xlabel='Paired difference (percentage points)',
        title='B  Participant contrasts',xlim=(-5.5,9.1))
    ax.invert_yaxis();ax.axvline(0,color='#77838B',ls='--',lw=1)
    for i,row in enumerate(rows):ax.text(6.0,i,f"p = {float(row['p_holm']):.3f}",fontsize=8,va='center')
    ax.text(.98,-.16,'Holm-adjusted p',transform=ax.transAxes,ha='right',fontsize=8)
    save(fig,output,'input_replacement')
    names=['haufe_rating1','haufe_rating4','layercam_rating1','layercam_rating4']
    labels=['Haufe\nlow','Haufe\nhigh','Layer-CAM\nlow','Layer-CAM\nhigh']
    fig,axes=plt.subplots(1,2,figsize=(9,4.5),layout='constrained')
    for ax,file,metric,title in [(axes[0],'haufe_layercam_overlap_frozen.csv','dice','A  Top-20% Dice overlap'),
                                 (axes[1],'haufe_layercam_correlation_frozen.csv','pearson_r','B  Continuous-map Pearson r')]:
        matrix=np.eye(4)
        for row in tables[file]:
            a=row['left_map'].replace('layercam_tfce_t','layercam').replace('layercam_groupmean','layercam')
            b=row['right_map'].replace('layercam_tfce_t','layercam').replace('layercam_groupmean','layercam')
            i,j=names.index(a),names.index(b);matrix[i,j]=matrix[j,i]=float(row[metric])
        ax.imshow(matrix,cmap='Blues' if metric=='dice' else 'RdBu_r',vmin=0 if metric=='dice' else -1,vmax=1)
        for i in range(4):
            for j in range(4):ax.text(j,i,f'{matrix[i,j]:.3f}',ha='center',va='center',fontsize=9,
                color='white' if abs(matrix[i,j])>.6 else '#1C2630')
        ax.set(xticks=range(4),yticks=range(4),xticklabels=labels,yticklabels=labels,title=title)
        ax.tick_params(length=0,labelsize=8)
    save(fig,output,'haufe_layercam_comparison')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference',type=Path,default=REFERENCE)
    parser.add_argument('--output-dir',type=Path,default=ROOT/'outputs/figures')
    args=parser.parse_args()
    if args.output_dir.resolve().is_relative_to(args.reference.resolve()):
        raise ValueError('Generated figures must not overwrite the reference archive')
    args.output_dir.mkdir(parents=True,exist_ok=True)
    tables=reproduce(args.reference)
    performance(tables,args.output_dir);regional(tables,args.output_dir);supplement(tables,args.output_dir)
    (args.output_dir/'figure_sources.json').write_text(json.dumps({
        'scope':'Table-driven redraws; no cortical surface generation or model rerun',
        'decoding_performance':'OOF participant BA; frozen participant-bootstrap 95% intervals',
        'layercam_macroregions':'Group-first regional means, voxel-weighted macroregions, whole-brain mean=1',
        'layercam_selected_rois':'Ten selected rows of the complete group-first ROI table; display names cleaned only',
        'layercam_top5_rating1':'Top five complete-table ROI values in each original macroregion (low subjective food value)',
        'layercam_top5_rating4':'Top five complete-table ROI values in each original macroregion (high subjective food value)',
        'input_replacement':'Recomputed BA; frozen bootstrap intervals and Holm-adjusted inference',
        'haufe_layercam_comparison':'Frozen Dice from ranked TFCE-masked t maps; frozen correlation from continuous group CAM maps'
    },indent=2)+'\n')
    (args.output_dir/'captions.md').write_text(CAPTIONS)
    print(f'Saved seven PNG/PDF figure pairs in {args.output_dir}')

CAPTIONS = """# Table-driven figure captions

- **decoding_performance:** Participant-average balanced accuracy for CNN and matched linear SVM at k = 1, 3, and 6 (30 participants). Error bars are archived participant-bootstrap 95% confidence intervals; points are recomputed from saved OOF predictions. Dashed line: 50% chance.
- **layercam_macroregions:** Voxel-weighted mean Layer-CAM values in the five original macroregion groups, divided by the whole-brain mean of the same category's k = 6 group map. The dashed line marks 1. The display label Insular/subcortical corresponds to the archived Subcortical group; the archived grouping is preserved.
- **layercam_selected_rois:** Selected k = 6 regional values, computed after participant/category maps were averaged within each category. Low and high subjective food value correspond to ratings 1 and 4. R/L denote right/left.
- **layercam_top5_rating1 / layercam_top5_rating4:** Five ROIs with the highest normalized mean Layer-CAM value in each original macroregion, ranked separately for low and high subjective food value. Values use the complete ROI table; label cleanup changes display text only.
- **input_replacement:** A: Participant-average BA after fixed k = 6 input replacements. Random-mask performance is averaged across 100 replicates within each participant before the group mean. B: The four prespecified paired contrasts with archived 95% bootstrap intervals and Holm-adjusted sign-flip p values. All confidence intervals and inferential p values are frozen archived estimates.
- **haufe_layercam_comparison:** Archived pairwise comparisons of k = 1 SVM Haufe patterns and k = 6 Layer-CAM maps. A: Dice overlap between top-20% selections; Layer-CAM selection uses the existing TFCE-masked t maps. B: Pearson correlation of continuous maps; Layer-CAM uses class-specific group mean attribution maps.

These are table-driven redraws. Original cortical surface figures are retained separately and are not regenerated by this command.
"""

if __name__=='__main__':main()
