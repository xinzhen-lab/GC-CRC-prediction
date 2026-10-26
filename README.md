# AI-based Ovarian Cancer Prediction with Laboratory Tests

This tool is developed and validated based on the data of three cohorts (with 8000+ positive cases vs. 6000+ controls) from hospitals in China: the First Affiliated Hospital of Sun Yat-sen University (SYSU-1), the Seventh Affiliated Hospital of Sun Yat-sen University (SYSU-7) and Shenzhen People's Hospital (SZPH). The internally validated AUC (SYSU-1 cohort) was 0.949 while the AUCs on the independent cohorts (SYSU-7 & SZPH) were respectively 0.926 and 0.905, indicating good discriminative ability. This model builds upon and extends a multi-classifier fusion methodology [1]. Other investigators are invited to share their data in order to further improve the generalization capability of the current model. The model should only be used to support clinical judgment by health care professionals as a complementary tool for screening Gastric cancer (GC) and colorectal cancer (CRC) with laboratory tests. Any responsibility for using this model and its results will rest solely by the health care professional using the model. Using it you should understand and agree that this tool is not responsible or liable for any claim, loss, or damage resulting from its use. While we try to keep the information on the tool as accurate as possible, we disclaim any warranty concerning its accuracy, timeliness, and completeness, and any other warranty, express or implied, including warranties of merchantability or fitness for a particular purpose.

# Study overview
<img width="1548" height="1068" alt="image" src="https://github.com/user-attachments/assets/ba8b976d-4b4f-4cbc-a625-6133e4789943" />


# Reference

[1] Q He, X Li, DWN Kim, X Jia, X Gu, X Zhen, L Zhou: Feasibility study of a multi-criteria decision-making based hierarchical model for multi-modality feature and multi-classifier fusion: Applications in medical prognosis prediction. Information Fusion 2020, 55:207-219.(https://doi.org/10.1016/j.inffus.2019.09.001)

# Model training

The pretrained model is stored in 'CS_MCF.model' that was trained on 10,449 (GC n=2,408, CRC n=3,938, Control n=4,103) patients' lab tests from SYSU-1 in China.

# Model validation

This tool has been validated on two external validation sets, including 961 (GC n=206, CRC n=210, Control n=545) patients' lab tests from SYSU-7, and 3,238 (GC n=375, CRC n=1,006, Control n=1,857) patients' lab tests from SZPH in China.


# Requirements

Python-3.8

joblib==0.14.2

six==1.16.0

scikit-learn==1.3.0

tenacity==8.2.3

threadpoolctl==3.5.0

typing-extensions==4.11.0

wheel==0.43.2

xgboost==2.11

zipp==3.17.0

plotly==5.22.0

pyparsing==3.0.9

python-dateutil==2.9.0post0

pytz==2024.1

packaging==24.1

kiwisolver==1.4.4

lightgbm==4.5.0

catboost==1.2.3

colorama==0.4.6

cycler==0.11.0

fonttools==4.51.0

graphviz==2.50.0

importlib-metadata==7.0.1

Pillow==10.4.0

# Usage

Make sure that all the required packages are installed, unzip 'CS-MCF.rar', run the 'CS-MCF code github version.py'to check.Input target laboratary items into 'data_eg.xlsx' . If all required packages are properly installed, it will give the Negative and Positive probabilities.

# Note

Missing values please leave blank. This model is tolerant of missing values. Derived composite indices is automatly calculated through processing.model. Note that more accurate predictions might be expected only when sufficient lab test items are provided.

This code is still under development and periodic updates will be taken to resolve the bugs.

The CS-MCF code github version.py contains codes for model training and validation. The current version is not the final version and may have bugs. Continue developement is ongoing and we will periodically update this file.

