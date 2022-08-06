<p align="center">
  <img alt="⚙️WGAN_for_MoG" src="https://user-images.githubusercontent.com/62103572/183245692-97c1607d-08a6-47df-a38a-508648112807.png">
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/y/EliaFantini/WGAN-learns-the-distributon-of-a-MoG">
  <img alt="GitHub last commit" src="https://img.shields.io/github/last-commit/EliaFantini/WGAN-learns-the-distributon-of-a-MoG">
  <img alt="GitHub code size" src="https://img.shields.io/github/languages/code-size/EliaFantini/WGAN-learns-the-distributon-of-a-MoG">
  <img alt="GitHub repo size" src="https://img.shields.io/github/repo-size/EliaFantini/WGAN-learns-the-distributon-of-a-MoG">
  <img alt="GitHub follow" src="https://img.shields.io/github/followers/EliaFantini?label=Follow">
  <img alt="GitHub fork" src="https://img.shields.io/github/forks/EliaFantini/WGAN-learns-the-distributon-of-a-MoG?label=Fork">
  <img alt="GitHub watchers" src="https://img.shields.io/github/watchers/EliaFantini/WGAN-learns-the-distributon-of-a-MoG?label=Watch">
  <img alt="GitHub star" src="https://img.shields.io/github/stars/EliaFantini/WGAN-learns-the-distributon-of-a-MoG?style=social">
</p>


Implementation of a Wasserstein Generative Adversarial Network that learns the distribution of a Mixture of Gaussian. The WGAN loss is Lipschitz constrained, so to enforces the constraint I implemented and tested two possible methods: weight clipping and Spectral Normalization. Since the
state-of-the-art GAN training is computationally expensive, this project will use a simple GAN with a linear generator and a dual variable.


For a more detailed explanation of the terms mentioned above, please read *Exercise instructions.pdf*, it contains also some theoretical questions answered in *Answers.pdf* (handwritten). 

The project was part of an assignment for the EPFL course [EE-556 Mathematics of data: from theory to computation](https://edu.epfl.ch/coursebook/en/mathematics-of-data-from-theory-to-computation-EE-556). The backbone of the code structure to run the experiments was already given by the professor and his assistants, what I had to do was to implement the core of the optimization steps, which are the FO and proximal methods algorithms and other minor components. Hence, every code file is a combination of my personal code and the code that was given us by the professor.

The following GIFs shows the output of the code **train.py**. The first one (left) is obtained by using the weight clipping to enforce a Lipschitz constraint, while the second one (right) is the result of using Spectral Normalization. 

<p align="center">
  <img width="300" alt="weight clipping" src="https://user-images.githubusercontent.com/62103572/183261924-41a9eed6-808e-4617-a4bc-f3ab1791083e.gif">
  <img width="300" alt="spectral normalization" src="https://user-images.githubusercontent.com/62103572/183261653-d9bf0d00-1b4d-4417-b532-45a71729927d.gif">
</p>


From those results, we can see how with both techniques the noise (red points) tends to have the same distribution as real data (blue points) and they keep oscillating between a less precise solution and a more accurate one. Using spectral normalization makes the noise achieve a similar distribution to the real data one after 600 iterations, while weight clipping is faster and achieves it after about 300 iterations. Spectral normalization does also get worse after 800 iterations and then gets back to a good distribution after 1000, whereas weight clipping maintains a more constant result.

## Author
-  [Elia Fantini](https://github.com/EliaFantini)

## How to install and reproduce results
Download this repository as a zip file and extract it into a folder
The easiest way to run the code is to install Anaconda 3 distribution (available for Windows, macOS and Linux). To do so, follow the guidelines from the official
website (select python of version 3): https://www.anaconda.com/download/

Additional package required are: 
- pytorch
- matplotlib
- tqdm
- imageio

To install them write the following command on Anaconda Prompt (anaconda3):
```shell
cd *THE_FOLDER_PATH_WHERE_YOU_DOWNLOADED_AND_EXTRACTED_THIS_REPOSITORY*
```
Then write for each of the mentioned packages:
```shell
conda install *PACKAGE_NAME*
```
Some packages might require more complex installation procedures (especially [pytorch](https://pytorch.org/)). If the above command doesn't work for a package, just google "How to install *PACKAGE_NAME* on *YOUR_MACHINE'S_OS*" and follow those guides.

Finally, run **train.py**. To use weight clipping instead of Spectral normalization, you have to change the line of code 110 in *trainer.py* from "f.enforce_lipschitz()" to "f.enforce_lipschitz(spectral_norm=False)".
```shell
python train.py
```

## Files description

- **code/src/** : folder containing all the sub-components useful for the training

- **code/train.py**: main code to run the training and create the result's GIF

- **code/movie.gif**: ouput of the train.py function (using weight clipping)

- **Answers.pdf**: pdf with the answers and plots to the assignment of the course

- **Exercise instructions.py**: pdf with the questions of the assignment of the course

## 🛠 Skills
Python, Pytorch, Matplotlib. Machine learning, Wasserstein Generative Adversarial Network (WGAN) implementation, minimax problems, implementation of both weights clipping and Spectral Normalization for Lipschitz constrained minimization.
## 🔗 Links
[![portfolio](https://img.shields.io/badge/my_portfolio-000?style=for-the-badge&logo=ko-fi&logoColor=white)](https://github.com/EliaFantini/)
[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/-elia-fantini/)
