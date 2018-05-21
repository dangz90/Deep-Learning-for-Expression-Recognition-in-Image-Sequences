# Deep Learning for Expression Recognition in Image Sequences

Facial expressions convey lots of information, which can be used for identifying emotions. These facial expressions vary in time when they are being performed. Recognition of certain emotions is a very challenging task even for people. This thesis consists of using machine learning algorithms for recognizing emotions in image sequences. It uses the state-of-the-art deep learning on collected data for automatic analysis of emotions. Concretely, the thesis presents a comparison of current state-of-the-art learning strategies that can handle spatio-temporal data and adapt classical static approaches to deal with images sequences. Expanded versions of CNN, 3DCNN, and Recurrent approaches are evaluated and compared in two public datasets for universal emotion recognition, where the performances are shown, and pros and cons are discussed.

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
* [Theano](http://deeplearning.net/software/theano/) - Theano is a Python library that allows you to define, optimize, and evaluate mathematical expressions involving multi-dimensional arrays efficiently.

## Author

* **Daniel Garcia Zapata** [[dangz90]](https://github.com/dangz90)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* The face frontalization [2] preprocess was performed using Douglas Souza [[dougsouza]](https://github.com/dougsouza/face-frontalization) implementation.
* The 3D CNN model is based on Alberto Montes [[albertomontesg]](https://gist.github.com/albertomontesg/d8b21a179c1e6cca0480ebdf292c34d2) implementation of C3D model.
* The CNN model is based on Refik Can Malli [[rcmalli]](https://github.com/rcmalli/keras-vggface) implementation of the VGG-Face.
* The [VGG-Face](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/) was first introduced by Omkar M. Parkhi, Andrea Vedaldi, Andrew Zisserman from University of Oxford.
* The [C3D Model](https://www.cv-foundation.org/openaccess/content_iccv_2015/html/Tran_Learning_Spatiotemporal_Features_ICCV_2015_paper.html) first introduced by Du Tran, Lubomir Bourdev, Rob Fergus, Lorenzo Torresani, Manohar Paluri
from Facebook AI Research and Dartmouth College.

## Bibliography
* [1] Ofodile, I., Kulkarni, K., Corneanu, C. A., Escalera, S., Baro, X., Hyniewska, S., ... & Anbarjafari, G. (2017). Automatic recognition of deceptive facial expressions of emotion. arXiv preprint arXiv:1707.04061.
* [2] Hassner, T., Harel, S., Paz, E., & Enbar, R. (2015). Effective face frontalization in unconstrained images. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (pp. 4295-4304).
