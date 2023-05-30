[![Discord](https://badgen.net/badge/icon/discord?icon=discord&label)](https://discord.gg/JvMQgMkpkm)
[![Compete on EvalAI](https://badgen.net/badge/compete%20on/EvalAI/blue)](https://eval.ai/web/challenges/challenge-page/2077/overview)
[![Win $1000](https://badgen.net/badge/win/%241%2C000.00/yellow)](http://mmsports.multimedia-computing.de/mmsports2023/challenge.html)

# The DeepSportradar Cricket Bowl Release Challenge (2023)


Welcome to the first edition of the DeepSportradar Cricket Bowl Release Challenge, which is one of the [ACM MMSports 2023 Workshop](http://mmsports.multimedia-computing.de/mmsports2023/index.html) challenges. 
An opportunity to publish, as well as winning a $1000 prize by competing on [EvalAI](https://eval.ai/web/challenges/challenge-page/2077/overview). 
See [this page](http://mmsports.multimedia-computing.de/mmsports2023/challenge.html) for more details.
In this challenge, participants will have to segment the input video to recognize the bowl release action.

## NOTE: The challenge dataset will be released in the next two weeks.

The dataset is split into a training, test and challenge set.
This challenge aims at segmenting the parts of the videos where a bowl release is happening.
Differnetly from other action detection challenges, here we are interested in detecting the time window when the bowl release action is happening.
The event lasts usually around 100 frames in the video. The objective of this challenge is to detect all bowl release events.

    
Maintainers: Davide Zambrano (d.zambrano@sportradar.com) from Sportradar.


&nbsp;
<p align="center"><img src="assets/banner.png" width="740"></p>

## Installation

**Note that the file ```setup.py``` specifies the libraries version to use to run the code.**

Install [PyTorch](http://pytorch.org/). 

```shell
git clone https://github.com/DeepSportradar/cricket-bowl-release-challenge/tree/v0.0.0
cd cricket-bowl-release-challenge
pip install -e .
```

## Example

Run the following command to train a baseline model:
```shell
python main.py --epochs 50
```

## Data

TBD

## Submission on EvalAI
Submit your result through the [challenge page on the EvalAI platform](https://eval.ai/web/challenges/challenge-page/2077/overview).


Please refer to the challenge webpage for complete rules, timelines and awards: [https://deepsportradar.github.io/challenge.html](https://deepsportradar.github.io/challenge.html).

## Questions and remarks
If you have any question or remark regarding the challenge and related materials, please raise a GitHub issue in this repository, or contact us directly on [Discord](https://discord.gg/JvMQgMkpkm).
