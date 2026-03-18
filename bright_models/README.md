# **Codebase to train BRIGHT models**
The organization was taken from the excellent [EDS-PSEUDO repository](https://github.com/aphp/eds-pseudo/tree/main).

The goal is essentially to train a similar model to EDS-PSEUDO but on much more fields to identify and the final use case will not be pseudonymization but feature extraction.

Currently files have been copied from the EDS-PSEUDO repository and they will need major adaptations/extensions to fit the BRIGHT project needs.

# **Plan**
This is a separate codebase for maintainability but the goal is to use this code base in order to train models that will be then used in the actual `src` of the `BRIGHT_ClinicalDataBase` repository. Most of the work will be to define the pipelines and rules for our 111 fields compared to the dozen fields the original EDS-PSEUDO was made for. This implies a good structure for the code and a good understanding of the original EDS-PSEUDO code base. Each semantic group should have its precise well defined pipes and rules so integration in how to train the models can be as close as it was for the original repo. The second big aspect is the dataset generation. In `../../test_annotaed` we have access to 17 pseudonymized real documents from diverse practitioners in the BRIGHT team and as well as the annotations so this should guide us to define the updates to the `generate_dataset.py` file. Our final dataset should be at least 500 documents long.

1) Prepare `generate_dataset.py`: The main difference here is that we have many more fields to generate. While this makes the code more complex, the goal remains to produce documents plausible enough for model training. (Since the public EDS-PSEUDO model was trained on synthetic data with great success, we will extend their methodology). If manual generation proves too complex, we will utilize LLM-based generation using the pseudonymized documents currently at our disposal. 
>DONE = LLM based generation, 4 step pipeline profile generation -> document generation -> span resolution -> clinical coherence check and other quality checks
> plus, added 600 documents with variations (typos, more variety)

2) Adapt `train.py`: Modify the training script to train EDS-PSEUDO-style models for each desired semantic group. A "semantic group" refers to a set of similar features; training them together helps the model learn to distinguish between closely related concepts. This phase also includes implementing a robust evaluation script for both training and testing sets.

3) Define Pipelines: For each field or group of fields, define the "pipes" as structured in the original repository to create specific training pipelines for each semantic group.

4) Deployment & Hosting: Develop the necessary code to launch training sessions on Google Colab. Additionally, include scripts to package model weights for Hugging Face. This is crucial for long-term maintenance and simplifies importing weights into the main `src` directory later.

5) Execution & Evaluation: Launch the training runs on Colab and perform thorough model evaluations.

6) Integration: If the performance is satisfactory (or at least superior to the current baseline), we will replace the existing pipeline (DateExtractor + ControlledExtractor + GlinerExtractor + EDSExtractor) with our custom BRIGHTExtractor. This new component will consist of various trained EDS models and trained span_linkers for fields with controlled vocabularies. A dedicated definition file for BRIGHTExtractor will be created in `../src/pipeline`.