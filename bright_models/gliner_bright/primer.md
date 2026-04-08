# Switching back to gliner
10 EDS NLP => Bad performance because focuses on the signal most present 
=> 65 EDS NLP => Better for individual ones, but still some really bad and 65 is just too much (23Gb)

=> ALthough out of the box gliner did not work well, a fine tuned set of gliner models might because: Finetuning on our generated documents and field/description pairs should mean a more precise mechanism for identification, especially if we finetune all linear blocks (i.e not only FFN but also the matrices in attention layers)
=> Could try 10 finetuned Gliner multi models, compare with previous results
=> Decide which strategy to go for (10 gliners ? slightly more than 10 gliners (max 20) to better handle some groups with conflicting precise semantics ?)

## HOW ?
Essentially go back to the 10 groups (should be alright because I did not delete them)
Need to make the training script for finetuning all linear of Gliner + train with field/description pairs ? => Need to look at the doc
As before also have script ready to prepare models for hugging face.
Visualizations for: Overall Precision / Recall / F1 comparison, Per feature Precision / Recall / F1 plots separately for both branches, Per feature groups plots. 
Ideally, have the script of training with two branches: EDS NLP or GLINER => Move old notebooks of EDS with 10 and 65 into a old_notebooks folder then Make a totally new, clean notebook in bright_models that launches training of models (per semantic groups we launch the EDS then the GLINER then evaluates them with a saved csv and the plot above), this for each semantic group: 
1. **diagnosis** — diag_histologique, grade, classification_oms, num_labo
2. **ihc** — 12 IHC markers (IDH1, ATRX, p53, Ki67, etc.)
3. **histology** — necrose, PEC, mitoses, aspect_cellulaire
4. **molecular** — 16 molecular genes (IDH1/2, MGMT, TERT, CDKN2A, etc.)
5. **chromosomal** — 9 chr arms + amplifications + fusions (17 labels)
6. **demographics** — sexe, birth year, IK, dominance, care team doctors
7. **tumor_location** — laterality, position, surgical location
8. **treatment** — chimios, protocols, surgery, RT, cycles, adjuvants (17 labels)
9. **symptoms_evolution** — epilepsy, deficit, cognitive, evolution, radiology (18 labels)
10. **dates_outcomes** — context-aware date labeling + survie_globale + infos_deces 


## TODO
MUST KEEP THE CODE SIMPLE !! the current code for eds_bright, scripts and notebooks was a bit of a mess and too complicated, it was moved to old_code
1) Read the official tutorials from GLINER repo => gliner_bright/tuto-*.md
2) Make the script to train gliner properly using the generated data (generated_data/*.jsonl) and to add entity descriptions (see src/extraction/gliner_extraction) during training. => bright_models/gliner_bright/training_gliner.py
3) Make the script for EDS NLP training. => bright_models/eds_bright/training_eds.py (can take some inspiration from the old_code/eds_bright code but must be more simple, the old way was a bit of a mess)
4) Make the unifiying script between GLINER and EDS NLP to have a unified interface AND output of models and results for validation/evaluation => bright_models/training.py
5) Make the visualizations and table saving utilities (should work for both EDS and GLINER) => bright_models/utils.py bright_models/viz.py
6) Make the 10 semantic group notebooks for colab (shoudl setup, prepare training, for each semantic group train and save results for GLINER and EDS and comparisons, once all semantic groups have been trained and results saved, make the overall plots for comparisons and individual performance assesment) => bright_models/notebooks/ten_groups_training.ipynb (some inspiration for the setup can be taken in the old_code/notbeooks but only for the setup) 
