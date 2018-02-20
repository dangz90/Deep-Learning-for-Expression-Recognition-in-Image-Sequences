# Deep Learning for Multi-Modal Hidden Emotion Analysis

A blocked emotion is one where the person is trying not to express it or tries to hide it despite feeling it. A recognition of that types of emotions is a very challenging task even for people. This thesis consists of using machine learning algorithms for recognizing blocked emotions. The thesis also considers creating a database, which would provide data for 5 basic emotions (Happiness, Surprise, Anger, Disgust and Sadness). The thesis uses state of the art deep learning on multi-modal collected data (video and EGG) for automatic analysis of hidden emotion. All the experiments uses SASE-FE dataset [1] for fake-real emotions of a facial expressions that are either congruent or incongruent with underlying emotion states. 

<!-- ## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

What things you need to install the software and how to install them

```
Give examples
``` -->

## Built With

* [Keras](https://keras.io/) - Keras is a high-level neural networks API, written in Python
* [Tensorflow](https://www.tensorflow.org/) - TensorFlow™ is an open source software library for numerical computation using data flow graphs.

## Author

* **Daniel Garcia Zapata** [dangz90](https://github.com/dangz90)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* The face frontalization [2] preprocess was performed using Douglas Souza [dougsouza] implementation (https://github.com/dougsouza/face-frontalization)
* The 3D CNN model is based on Alberto Montes [albertomontesg] implementation of C3D model. (https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2)
* The CNN model is based on Refik Can Malli [rcmalli] implementation of the VGG-Face (https://github.com/rcmalli/keras-vggface)
* The VGG-Face was first introduced by Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman from University of Oxford
(http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)
* The C3D first introduced by Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, Manohar Paluri
from Facebook AI Research and Dartmouth College 
(https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html)

## Bibliography
* [1] Ofodile, I., Kulkarni, K., Corneanu, C. A., Escalera, S., Baro, X., Hyniewska, S., ... & Anbarjafari, G. (2017). Automatic recognition of deceptive facial expressions of emotion. arXiv preprint arXiv:1707.04061.
* [2] Hassner, T., Harel, S., Paz, E., & Enbar, R. (2015). Effective face frontalization in unconstrained images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4295-4304).
