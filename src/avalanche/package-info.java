/**
 * Super high-level machine learning library
 *
 * @since 1.14.10
 * Supports:
 *    - K-Means clustering
 *    - Hierarchical clustering
 *    - Neural nets
 *       (Multi-Layer Perceptron, will eventually support
 *        Convolutional Neural Nets and Recurrent Neural Nets)
 *    - Simple genetic evolution algorithm
 *       (Currently fixed-topology, however NEAT is planned)
 *    - A truly watered-down implementation of matrices
 *    - Q-Learning
 *    - Simple Linear Regression
 *    - Multiple Linear Regression **EXPERIMENTAL**
 *
 * Planned features (If a feature has been implemented, the annotation
 *                   will be @version not @since and the correct release number)
 *
 * (The @version javadoc annotations are only rough estimates):
 *  @version 1.9.9   - Ability to toggle system logging
 *  @version 1.15.1  - Adjusted r-squared, p, t, etc.
 *  @version 2.0.0   - Expectation-maximization clustering with mixture
 *  @version 2.1.1   - Logistic regression
 *
 *  @version 2.1.2   =====| BETA    + DEBUG |=====
 *
 *  @version 2.2.0   - Convolutional neural nets (pooling and convolution)
 *  @version 2.x.0   - Generative Adversarial Nets
 *  @version 2.3.0   - Dendrogram graphing
 *
 *  @version 2.4.0   =====| RELEASE + DEBUG |=====
 *
 *  @version 2.5.9   - Batch normalization
 *  @version 2.6.0   - NEAT genetic algorithm
 *  @version 2.7.0   - Sparsely connected nets
 *  @version 2.7.x   ==[ Polish everything up, debug everything ]==
 *  @version 2.8.0   - Reinforcement learning
 *  @version 2.9.0   - Recurrent neural nets
 *  @version 3.0.0   - Deep-belief nets
 *  @version 3.1.0   - N-Dimensional arrays
 *  @version 3.1.x   ==[ Polish everything up, debug everything ]==
 *  @version 3..x.x  - More to come
 *
 *
 *
 * @author Henry Zhang
 *
 * Real release history (the topmost version is the most current)
 * @version 1.15.0      November 4, 2017 (2017.11.4)
 *   Multiple Linear Regression somewhat implemented, much work is still to be done however.
 *   Correlation Dataset Made
 *
 * @since 1.14.10       October 21, 2017 (2017.10.21)
 *   Matrix inversion and determinant calculation implemented
 *
 * @since 1.13.0        October 15, 2017 (2017.10.15)
 *   Simple Linear Regression
 *
 * @since 1.12.1        October 14, 2017 (2017.10.14)
 *   Fixed critical ClusterUtils bug
 *
 * @since 1.11.0        October 14, 2017 (2017.10.14)
 *   Made separate Dataset class for storing data, changed Hierarchical and KMeans mechanics
 *   Made Clustering class with many helper functions
 *
 * @since 1.10.0        October 12, 2017 (2017.10.12)
 *   Hierarchical clustering with no auto-abort for clustering
 *
 * @since 1.9.1         October 8, 2017 (2017.10.8)
 *   package-info.java made
 *
 * // TODO and FIX MEs
 * FIXME: Multiple linear regression, why is b=((X′X)^−1)X′y supposed to work (because it doesn't)
 * TODO: Pooling and conv.ing mutliple channels without combining them into one channel
 * TODO: Standardized coefficients, std. error, deg of freedom, average residual, t & p value, etc etc etc.
 * TODO: No doubles, use BigDecimals
 * TODO: Restricted Boltzmann Machines
 * TODO: fork (Pooling and Convolution Layer)
 */
package avalanche;