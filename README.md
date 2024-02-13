# OCR-model

In the process of developing this project, I opted to utilize a Convolutional Neural Network (CNN) due to its emulation of human-like visual perception. Much like how humans recognize patterns and details in images, CNNs excel at capturing intricate spatial relationships and features within visual data. This choice was driven by the network's capacity to automatically learn and discern hierarchical patterns, making it well-suited for tasks like character detection. By leveraging the strengths of CNNs, the application achieves a level of image understanding that closely mirrors human visual processing, enhancing its capability to accurately extract letters from diverse and complex image backgrounds. This human-like approach to image recognition contributes to the overall effectiveness and adaptability of the application.

About Dataset:
API Command : kaggle datasets download -d vaibhao/handwritten-characters
https://www.kaggle.com/datasets/vaibhao/handwritten-characters

Context:
This dataset is based on the EMNIST data for Alphabets and Digits, with additional special characters (@, #, $, &). The images have been transformed using various image processing techniques, resulting in 32x32 pixel black and white images. The categories have been carefully merged to prevent misclassification, totaling 39 categories in the training and validation sets.

Content:
The dataset encompasses English alphabets (both lowercase and uppercase), digits (0-9), and selected special characters (@, #, $, &). The images are standardized to 32x32 pixels, presented in black and white. The categories include 26 for alphabets (combining lowercase and uppercase), 9 for digits (1-9), and a combined category for digit 0 and character O. The dataset is ideal for applications involving handwritten text, such as OCR development.
