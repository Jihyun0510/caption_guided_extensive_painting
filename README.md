Captioning-Based Extensive Painting
=============
 We developed a new approach for image outpainting and wide-range image blending, which we refer to as *extensive painting* for convenience, implementing a novel modality of hints, **text**. In order to incorporate this novel type of hints to extensive painting, we propose a Captioning-based Extensive Painting (CEP) module, which consists of two multi-modal tasks, an image captioning task and a text-guided image manipulation task. The image captioning model generates caption of an input masked image, and the text-guided image manipulation model fills in the unknown region with the guidace of the text hints.

<img width="100%" src="./images/main_concept.png"/>

* Image captioning models to be found in image_captioning folder.
* Text-guided image manipulation models to be found in image_manipulation folder.
