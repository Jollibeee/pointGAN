# EMD variants for 3D Point Clouds

This repository builds on top of the [optas](https://github.com/optas/latent_3d_points) repository. That code base is created
by [Panos Achlioptas](http://web.stanford.edu/~optas/), [Olga Diamanti](http://web.stanford.edu/~diamanti/), 
[Ioannis Mitliagkas](http://mitliagkas.github.io), and [Leonidas J. Guibas](http://geometry.stanford.edu/member/guibas).
Their work is based on an [arXiv tech report](https://arxiv.org/abs/1707.02392). 

It is a deep net architecture for auto-encoding point clouds. Check also the [project webpage](http://stanford.edu/~optas/).

## Citation

You can cite original authors by:

	@article{achlioptas2017latent_pc,
	  title={Learning Representations and Generative Models For 3D Point Clouds},
	  author={Achlioptas, Panos and Diamanti, Olga and Mitliagkas, Ioannis and Guibas, Leonidas J},
	  journal={arXiv preprint arXiv:1707.02392},
	  year={2017}
	}

## Observation

The reason to look further into this type of network is when we start to use more geometric objects. If we morph from
one car to the next, all the intermediate steps do visually make sense. The function from transportation theory, the
EMD or Wasserstein distance, seems to be appropriate. Now, in which cases does this lead to limitations?

* When multiple objects have to be recognized. 
* When we have objects of a sparses nature, such as wireframes.

A very challenging dataset for this type of 3D point clouds is - at first sight - very simple. Cubes in a 3D space
where points are distributed over the edges of the cubes. 

This dataset can be found at:

* [Dataset cubes](https://github.com/mrquincle/dataset-cubes)

When you sweep over the latent representation that is formed while the autoencoder is being trained on the cube dataset,
it is very apparent that it did NOT form a proper representation. When going from one representation of a training
sample to the next, the intermediate representations do not form cubes at all. The points become one large 
spread out cloud upon which they "become" in the end something resembling cubes again. Look at this animation on 
[Twitter](https://t.co/xCtyJYvR9x) which makes this immediately clear.

![Cube marching on](https://gist.githubusercontent.com/mrquincle/35a9d1e8a12bf86c0d059283611fe281/raw/f1b42226a2ea81267a51c114f1293b07cdbb44fb/cubes_emd_latent_sweep.gif)

We need a more sophisticated form of optimal transport. A form of transport that can be compared with moving dirt 
from one location to the next (as with ordinary earth mover's distance), but where we use trucks. We first move 
everything from a particular location into a truck. Then we go to a particular destination and unload the truck. We
only count manual labor and disregard the fuel that goes into the truck.

That's the basis of an adaptation to the EMD algorithm that can be found at:

* [EMD suite](https://github.com/mrquincle/emd-suite)

Note that the parallel with trucks is not exact. 

1. If trucks would really drive for "free", we would throw only one piece of dirt into the truck and let it drive to a
particular destination.
2. The loading and unloading disregards local structure. We want to preserve such structure. In other words, it is more
like we put a plate under the distribution of sandpile at a particular spot, move it in its entirety, and then unload it
without shifting the dirt around. Only when it arrives we reorganize using EMD. It's like we transport a car. We will
not completely disassemble it and assemble it again at the destination. We just transport it as a whole and at the
destination change tiny parts if required.

Let us describe how we change the autoencoder to experiment with this type of EMD.

## Adaptation

The adaptations to the original code base can be found in the [structural losses](https://github.com/mrquincle/latent_3d_points/tree/master/external/structural_losses)
directory. The [multiemd file](https://github.com/mrquincle/latent_3d_points/blob/master/external/structural_losses/tf_multiemd.cpp) implements
a variation on EMD, which is called multi-EMD. It calculates EMD over subsets of the data. The subsets right now are
established in a naive fashion. There is a `calc_offset` function in `sort_indices.c` which returns the offset over
subsets of points that are not further than a predefined distance apart. This distance is calculated between every point
in both sets. In practice this means that clusters of points will be compared. The resulting `offset` then identifies
the "center of mass" of those subsets. The offset is then subtracted from both point clouds and the normal EMD
is calculated.

To iterate faster with slightly different algorithms this is also implemented in Python in [point_net_ai](https://github.com/mrquincle/latent_3d_points/blob/master/src/point_net_ae.py). We can namely calculate such shifts before we invoke the CUDA optimized `approx_match` function. We have
experimented with:

* `emd`: original EMD implementation
* `multi_emd`: using the C/CUDA implementation
* `shift_emd`: same as `multi_emd`, but implemented in Python/Tensorflow
* `match_shift_emd`: as `shift_emd`, calculating the match values with clustered offsets, but calculating the costs using the non-shifted original point clouds
* `batch_shift_emd`: to speed things up, use batches (do not use, the sort function is now not global, which leads to artifacts)
* `lagrange_emd`: use a combination of normal EMD and `shift_emd` (this does not seem to work, they do not like each other's results).

The shifted version of the EMD strongly prefers a single object to be the result of the autoencoder. In other words, the
autoencoder is not so much an autoencoder. It encodes an object that can occur in multiple spatial locations. 

Its limitations:

* It only works on identical objects. 
* It currently has an internal (spatial) clustering mechanism that is not general.

Can it be used already? Yes, this autoencoder can be used as the first step to come up with a representation of
a particular (complex) object of which we have multiple copies. In other words, if finds a "prototype" of a
set of 3D objects. We can then in the second step feed this into a sampler. This sampler can those use the information 
of the "prototype" to find the location of the 3D objects. In other words, we separated a clustering task into an object
representation task and a positioning/tracking task.

The tracking itself is non-trivial:

* [Sampling code](https://github.com/mrquincle/noparama)

## Experiments

The experiments are done on Google Colab. For example using this [Notebook](https://colab.research.google.com/drive/1o7mnpigvnnyTGjNfa1emBb5A0M9Zsubm).

## Disclaimer

This is a work in progress. 

## License
This project is licensed under the terms of the MIT license (see LICENSE.md for details).
