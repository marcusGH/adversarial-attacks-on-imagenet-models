# Adversarial attacks on ImageNet models

This project explores adversarial attacks on ImageNet models such as Resnet.
Some of the key results and discoveries are briefly presented below. The full report
can be found in the [notebook](adverserial_attacks_imagenet.ipynb).

<!-- vim-markdown-toc GFM -->

* [FGSM](#fgsm)
* [Targeted attacks](#targeted-attacks)
* [Blackbox attacks](#blackbox-attacks)
* [Targeted blackbox attacks](#targeted-blackbox-attacks)
* [Universal targeted attacks](#universal-targeted-attacks)
* [References](#references)

<!-- vim-markdown-toc -->

## FGSM

By using the _fast gradient sign_ method (FGSM) described on page 3 of [(Szegedy et al, 2013)](https://arxiv.org/abs/1412.6572), we can quickly perturb an image of a dog to misguide RESNET-50 into thinking it's something
completely different.

![image](https://user-images.githubusercontent.com/29378769/195874464-b530851f-08e4-468b-ab85-b8a0f2827e3c.png)

## Targeted attacks

To do target attacks, we use gradient descent on

$$ \underset{\mathbf{\eta}}{\mathrm{argmin}} \;J(\theta,\mathbf{x}+\mathbf{\eta},\textrm{target label}), \textrm{ subject to 'small' } \eta,
$$

where $J$ is the cost function used to train RESNET-50 and $\epsilon$ is a hyperparameter for the amount of noise.
With this, we can for instance make the Border Collie dog seems like a sea slug:

![image](https://user-images.githubusercontent.com/29378769/195877904-47d9a347-4a1e-4462-af3d-4f04e9208990.png)

## Blackbox attacks

We can also perform blackbox attacks on the model (i.e. only using the forward pass of the network) by iteratively choosing a random direction $\eta \sim \mathcal{N}(\mathbf{0}, \sigma)$ and updating the noise on the image if this increases the model's loss. Doing this for just 200 iterations makes the model missclassify the dog breed, and after 10 000 iterations, the dog is perceived as toilet paper:

![image](https://user-images.githubusercontent.com/29378769/195879223-ae298711-2749-4057-b035-9327ef3ebc03.png)

## Targeted blackbox attacks

To do targeted blackbox attacks, we use a similar method to the above, but optimise with a target label in place
instead of just maximising the loss. As an example, I have done this with the dog image and the label `sea slug`.
To the left, I have plotted the prediction on the final image (20 000 iterations) and on the right, we see how
the predicition converges to sea slug and the correct prediction decreases in probability.

Prediction | Convergence
:---:|:---:
![image](https://user-images.githubusercontent.com/29378769/195880259-b8e25e8a-71d7-49e1-a91b-bed00b4675d9.png) | ![image](https://user-images.githubusercontent.com/29378769/195880203-8879293d-e67f-4f98-9a49-d55ba1e80b58.png)

## Universal targeted attacks

A universal attack is when we find a single noise vector and apply it to several different images to misguide
to model. To achieve this, I downloaded a subset of the ImageNet datasets, and trained a model to optimize
the following:

$$ \underset{\mathbf{\eta}}{\mathrm{argmin}} \frac{1}{n} \sum_{(\mathbf{x},l) \in \mathbf{X}} J(\theta, \mathbf{x}+\mathbf{\eta},\textrm{target label}),
$$

where $\mathbf{X}$ is a minibatch of training data (tuples of an image and the corresponding label). I held
out 12 000 test images and applied the noise derived during training to all of them. I then looked at whether
"sea slug" was the top prediction or among the top-5 predictions on these images. As we see below,
the model incorrectly thinks over 90% of the 12 000 images are sea slugs, even if the noise is the exact
same on all of these images.

Model | Loss | Top-1 Success rate | Top-5 Success rate
:------------:|:--------:|:----:|:------:
Resnet50 | 15.683664 | 	0.915417 | 	0.958417

We can also visualise how this unviversal noise looks like by applying it to a dog and a panda:

Dog | Panda
:----:|:----:
![image](https://user-images.githubusercontent.com/29378769/195882996-197370b8-3ccb-4d4f-bc33-b3689cf34f6e.png) | ![image](https://user-images.githubusercontent.com/29378769/195883062-33224fdc-5e1f-423e-b7e8-14e85dc81d15.png)

## References

Goodfellow, Ian J. and Shlens, Jonathon and Szegedy, Christian (2014). "Explaining and Harnessing Adversarial Examples", arXiv, https://doi.org/10.48550/arxiv.1412.6572
