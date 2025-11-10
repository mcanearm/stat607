I chose to reproduce Santander Greenland's paper: 

Greenland, Sander. “Methods for Epidemiologic Analyses of Multiple Exposures: A Review and Comparative Study of Maximum-Likelihood, Preliminary-Testing, and Empirical-Bayes Regression.” Statistics in Medicine 12, no. 8 (1993): 717–36. https://doi.org/10.1002/sim.4780120802.


* How well did you reproduce the original results? What differed?

I think I reproduced the original results somewhat faithfull. There were more tables used in the paper, whereas I chose to focus on visualizations. However, the biggest difference is that I seem to have a lot more outlier simulation runs, and I am not rejecting as many invalid simulation steps. I essentially cut out the highest 1% of my simulation estimtes for RMSE, and even then for some of my small sample sizes and small $n$, the plots are a bit wonky.

* Evaluate the neutrality of the simulation design

Now that I am more familiar with the paper, I think it's relatively neutral. After all, the original study begins with the MLE estimator, and then builds on it by implementing with new methods. I also think the paper does not push for any single outcome and seems to give each of the two alternative methodologies (empirical Bayes and semi-Bayes) equal weight.

* Were the simulation conditions fair to all methods being compared? Did they ignore important predecessors?
Yes, I think the simulations were fair. The data generating process focuses on something realistic for epidemiological data, namely a binary matrix of exposures and moderate correlation between the exposures. This makes sense for the domain, but this is a setting that should favor constrained coefficients. 

* Were there any design choices that favored certain methods?

By design, we know the true data generating process here has an effect of 0 roughly 80% of the time. We would expect that methods that regularize the coefficients vs. the MLE method should be better at detecting this difference, and so this favors empirical Bayes and semi Bayes methods over the default MLE method. That being said, the author is honest about times when the various methods outperform one another.

* Were realistic scenarios included, or only idealized conditions?

Yes, they were, specifically a real dataset analyzed at at the end, though I did not recreate this as part of my analysis. The author included a "null" estimator and preliminary testing method as well that aligns with standard epidemiological practice as a control. Those methods were also omitted.

* What would you change about the simulation design? Why?

I think the age of the paper means that using closed-form posterior solutions were required at the time. With large scale computation, that is less required, and so I would be curious to evaluate the performance against fully Bayesian methods in this case. Even though it takes more iterations and a sampler, I think in a relatively simple setting such as this one, it could be a really informative model and offer a nice contrast.

I would also add back in the two control methods, preliminary testing and the null estimator (just assuming zero). It would be nice to use those as a match against the paper.

* Did you recreate the visualizations or make your own? What did the visualizations reveal that the original paper missed or undersold? What was surprising or unexpected in the results?

I created my own visualizations and wasn't completely able to re-create the results. I talk about why in the future section. I think the original paper somewhat glosses over the difficulty of actually simulating data that doesn't have model fitting issues. The author talks about it, but it would have been helpful to see a table or visualization that connected model parameters to the overall success rate of the simulations. I store this information, but have yet to analyze it.

That being said, judging by RMSE of the $\beta$ values, I found an improvement of semi-Bayes methods in smaller samples compared to both the MLE and empirical Bayes. In medium samples, empirical Bayes seemed to do the best, but in large samples, the methods seem to converge
strongly in performance (N=2000).

* Which aspects of the implementation were most challenging?

I actually had the most difficulty making sure that I implemented the model fitting process correctly. It was challenging to read through the notation and then be confident that everything was working correctly. On a practical level, the hardest part was probably stopping progress to move forward. I actually thought everything was correct until I went to work on the visualizations, and then it became obvious that I was having some horrible RMSE values that indicated poor fits in some of my simulations. I would have spent more time, in retrospect, getting through the entire pipeline before perfecting earlier steps.

* How confident are you in your results? What could undermine that confidence?

I am not very confident in my results. What would improve my confidence is if the MLE values were all well-conditioned and the
semi-Bayes methods were a sort of "average" of the MLE and parametric bayes methods. I do have this to some degree, but I think it could be better. The number one indication that my simulation is currently off is the interval length, and so putting more effort into that particular metric would be benficial.