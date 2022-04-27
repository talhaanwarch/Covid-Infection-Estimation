# Covid-Infection-Estimation
This code is assosiated with the Covid-19-Infection-Percentage-Estimation-Challenge orgainzed at codalab. Details can be found [here](https://github.com/faresbougourzi/Covid-19-Infection-Percentage-Estimation-Challenge)
# Paper
## Title
SEnsembleNet: A Squeeze and Excitation based Ensemble Network for COVID-19 Infection Percentage Estimation from CT-Scans
## Abstract
Coronavirus (COVID-19) is a contagious disease caused by SARS-CoV-2 virus. Usually, COVID-19 is diagnosed by PCR test, which requires less human expertise, but this test's false-negative ratio is high. COVID can also be diagnosed from radiographs such as CT-scan and X-ray, but it requires expert radiologists. So there is a need for an automated way to interpret chest radiographs using artificial intelligence. Several labelled datasets and deep learning algorithms are available to diagnose corona patients using radiographs. These algorithms classify the images into predefined categories such as healthy or infected. But there is no way to know how much area of chest radiograph is infected by COVID. This paper proposed an ensemble network to predict COVID-19 percentage infection from a chest CT scan. The proposed ensemble network used squeeze and excitation bock to learn individual models' weights during the training process. On validation data and test data, the proposed approach obtained a mean absolute error of 4.469 and 3.64, respectively. Implementation is publicly available at https://github.com/talhaanwarch/Covid-Infection-Estimation

## Citation
```
@article{anwar2022sensemblenet,
  title={SEnsembleNet: A Squeeze and Excitation based Ensemble Network for COVID-19 Infection Percentage Estimation from CT-Scans},
  author={Anwar, Talha},
  year={2022},
  publisher={TechRxiv}
}
```
