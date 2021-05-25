This repository contains the code to generate the winning solution of the [Kaggle competition on indoor location and navigation organized by Microsoft Research](https://www.kaggle.com/c/indoor-location-navigation/).

Our team name: "Track me if you can".

Authors:
- Are Haartveit
- Dmitry Gordeev
- Tom Van de Wiele

![Ranking](https://i.ibb.co/KhzRZ72/final-ranking.png)

## References
- [Indoor location navigation Kaggle competition](https://www.kaggle.com/c/indoor-location-navigation/)
- [Summary of our winning solution](https://www.kaggle.com/c/indoor-location-navigation/discussion/240176)
- [Competition ranking](https://www.kaggle.com/c/indoor-location-navigation/leaderboard)

## Steps to obtain the approximate winning submission
1. Clone the repository, it doesn't matter where you clone it to since the source code and data are disentangled.
1. Create a project folder on a disk with at least 150GB of free space. Create a "Data" subfolder in your project folder. This will be referred to as "your data folder" in what follows.
1. Download the raw text data from [here](https://www.kaggle.com/c/indoor-location-navigation/data) and extract it into your data folder.
1. Download the cleaned raw data from [here](https://www.kaggle.com/tomokikmogura/indoor-location-navigation-path-files?select=train) and extract it into the "reference_preprocessed" subfolder of your data folder.
1. Add your data folder to line 19 in src/utils.py
1. Run main.py
  
If all goes well, the pipeline should create a "final_submissions" subfolder in your data folder with two final submissions. Note that these are likely slightly different from our actual submissions due to inherent training stochasticity. When you make a late submit of these submissions to the leaderboard, you should obtain a private score around 1.5, which can be further reduced to about 1.3 after fixing the private test floor predictions (not part of this repo).

## Main script parameters
- Mode ("-m" or "--mode"). Default: 'test'. Select from ('valid', 'test').
- Suppress multipricessing ("-s"). Default: no suppression of multiprocessing.
- Fast (and bad) sensor models ("-f"). Default: no fast sensor models. Mostly useful for verifying that all dependencies are in place. Ignored when copying sensor models (next parameter).
- Copy sensor predictions ("-c"). Default: no copying of pretrained sensor predictions. Useful if you want to speed up the pipeline since training sensor models is the slowest part.

## Hardware requirements
Due to the size of the data set, you need at least 32 GB RAM to be able to run the pipeline successfully.

## Known issues
- If you run out of memory, try running the pipeline again. It should continue where it left it in the previous run.
