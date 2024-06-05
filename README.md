# Fake-News-Detection

Domain-Specific Fake News Detection

## Installation Instructions
1. Install latest version of CUDA if running on GPU (RECOMMENDED):
https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html

2. Install latest version of Python:
https://www.python.org/downloads/

3. Install latest version of Python pip  
``curl https://bootst/rap.pypa.io/get-pip.py -o get-pip.py``  
``python get-pip.py``

4. Install jupyter notebook  
``pip install jupyter``

5. Clone repo locally  
``git clone https://github.com/howardt12345/Fake-News-Detection.git``

### Installing Torch with Cuda
1. Navigate to https://pytorch.org/get-started/locally/
   
3. Choose the OS, package, language(Python) and CUDA Version Appropriate to your machine
   
5. Copy the command and run in the chosen platform terminal
   
7. To check CUDA version run ``nvcc --version``
   
9. To check if CUDA is installed and working open Python interpreter, ``import torch`` and run ``torch.cuda.is_available()``

### To Run Data Pre-Processing ###
1. Navigate to ``Fake-News-Detection/preprocessing/<insert category/dataset _ Preprocessing>.ipynb``
   
3. Run all cells
   
5. View output in the notebook and feather file in the /data/ folder in base directory
   
### To Run Classification ###
1. Navigate to base directory
   
3. Run ``Python /classification/bert.py``
   
5. View output in the /classification/logs/ directory

### To Run Detection ###
This will train and test models

1. Navigate to ``Fake-News-Detection/detection/run_detection.ipynb``
2. Change parameters according to anticipated results
3. Run ``run_detection.ipynb`` and wait for completion
4. View log in category folder to see results, if I run ``politics`` models, I navigate to ``Fake-News-Detection/results/politics`` and find the log file ``training_log_DATE_TIME.log``

To analyze results  
Note: there must be a valid log file to read from

1. Navigate to ``Fake-News-Detection/detection/result_analysis.ipynb``
2. Run the notebook and view cell outputs



## Proposed Architecture

The fake news detection will have the following stages:
1. Categorization stage: Categorize the input and pass the input to the model trained for the specific domain
   - https://towardsdatascience.com/topics-per-class-using-bertopic-252314f2640
3. Detection stage: With models trained to detect fake news within a specific domain, detect whether the input is fake news or not


## Potential Datasets

- Politics: (both)
  - [x] LIAR Dataset: Contains statements made by political figures, labeled by their truthfulness. (https://huggingface.co/datasets/liar)
  - [x] ISOT Dataset (multipurpose dataset) (https://onlineacademiccommunity.uvic.ca/isot/2022/11/27/fake-news-detection-datasets/)
  - [x] Fake News Dataset (https://www.kaggle.com/competitions/fake-news/data)
  - [x] PolitiFact Dataset: Includes fact-checks of political statements. (https://www.kaggle.com/datasets/rmisra/politifact-fact-check-dataset) (https://github.com/KaiDMML/FakeNewsNet)
- Healthcare and Medicine: (one is enough - choose the larger)
  - [ ] HealthMisinfo Dataset: Focuses on COVID-19 misinformation in healthcare. (https://trec-health-misinfo.github.io/)
  - [ ] FakeHealth Dataset: Contains fake health-related articles. (https://github.com/EnyanDai/FakeHealth)
- Science and Technology: (one is enough - choose the larger)
  - [ ] BuzzFeedNews Dataset: Includes news articles from BuzzFeed News. (https://www.kaggle.com/code/sohamohajeri/buzzfeed-news-analysis-and-classification)
  - [ ] FakeTech: Contains technology-related fake news.
- Finance and Stock Markets: (one is enough - choose the larger)
  - [ ] Stock Market Fake News Dataset: Focuses on news related to stock markets.
  - [ ] Financial News Dataset: Contains financial news articles.
- Entertainment and Celebrity News: (one is enough - choose the larger)
  - [ ] CelebA Dataset: Contains celebrity images (for image-based fake news). (https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
  - [ ] Entertainment News Dataset: Focuses on entertainment news. 
- Sports: (one is enough - choose the larger)
  - [ ] SportFake Dataset: Contains sports-related fake news articles.
  - [ ] Sports News Dataset: Contains sports news articles.
- Environment and Climate Change: (one is enough - choose the larger)
  - [ ] Climate Change Misinformation Dataset: Focuses on climate change misinformation. (https://huggingface.co/datasets/climate_fever/viewer/default/test?row=0)
  - [ ] Environmental News Dataset: Contains news related to the environment. 
- Education:
  - [ ] Education Misinformation Dataset: Focuses on educational misinformation.
- Crime and Security: (FA-KES is important)
  - [x] Crime News Dataset: Contains news articles related to crime.
  - [x] FA-KES (https://zenodo.org/record/2607278)
- Travel and Tourism:
  - [ ] Travel Misinformation Dataset: Focuses on travel-related misinformation.
- Religion and Spirituality:
  - [ ] Religion Misinformation Dataset: Contains religious misinformation. (https://data.mendeley.com/datasets/5ykks3psks/5)
- Transportation:
  - [ ] Transportation Misinformation Dataset: Focuses on transportation-related misinformation.
- Social Media and Online Discussions:
  - [x] PHEME (https://figshare.com/articles/dataset/PHEME_rumour_scheme_dataset_journalism_use_case/2068650)
  - [x] GossipCop (https://github.com/KaiDMML/FakeNewsNet)
  - [x] Fake News Detection (https://www.kaggle.com/jruvika/fake-news-detection/version/1)
