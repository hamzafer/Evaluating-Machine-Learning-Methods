# Evaluating Machine Learning Methods

## Ongoing Work and Future Directions

Currently working on making the project's findings more accessible and impactful within the academic and professional communities. There are two main directions which the Professor Phill and I are working on:

1. **ICC Website Publication:** We are in discussions to potentially publish our project on the International Color Consortium (ICC) website. This would involve a detailed exposition of our methodologies, tailored for software developers involved in profile construction using regression-based methods. The final decision will be made following the upcoming ICC meetings, which will consider the project's fit for the 'Information on Profiles' section.

2. **Conference Publication:** We are also considering submitting our work for presentation at a prestigious conference, such as the Color and Imaging Conference (CCIW) in Milan. This would require a thorough review of existing literature to highlight the novel aspects of our work, possibly focusing on our comprehensive comparison of machine learning methods or the introduction of a new method showing significant improvements over traditional approaches. Depending on the outcomes of our initial submission to the ICC, we may prioritize conference publication to ensure the novelty and impact of our research are fully recognized.

## Introduction
This project investigates the effectiveness of various machine learning models in predicting XYZ color values from CMY input data, compared to traditional polynomial regression methods. Utilizing datasets such as FOGRA51, APTEC(PC10), and APTEC(PC11), the research explores the accuracy of machine learning techniques including Gradient Boosting and Random Forest.

## Objectives
- To bridge the gap in current literature by introducing a comparative analysis of modern printer datasets.
- To systematically compare a suite of machine learning algorithms to polynomial regression in color characterization tasks.

## Methodology
- **Data Collection:** Utilized datasets from the International Color Consortium.
- **Machine Learning Methods:** Bayesian Methods, Decision Trees, Deep Learning, Elastic Net, Gradient Boosting, k-Nearest Neighbors, Lasso Regression, Random Forest Regressor, Ridge Regression, Simple Multilayer Perceptron (MLP), and Support Vector Machine (SVM).
- **Polynomial Regression:** Implemented for degrees ranging from 1 to 10.
- **Evaluation:** Error metrics calculated using the CIEDE2000 formula.

## Results
Machine learning models demonstrated varying degrees of improvement in error metrics over polynomial regression. However, the improvements were relatively modest, suggesting that the benefits of machine learning must be carefully balanced against their complexity.
<img width="905" alt="image" src="https://github.com/hamzafer/ColorProject/assets/45764331/66fc77f5-97fe-408a-941e-d314d541cdf5">
<img width="905" alt="image" src="https://github.com/hamzafer/ColorProject/assets/45764331/2de89479-f5ad-4705-bc8a-4a95c2ac8b1f">
<img width="905" alt="image" src="https://github.com/hamzafer/ColorProject/assets/45764331/ee480a9b-51db-42f6-948e-0a9479b18612">

## Conclusion
The study highlights the potential of machine learning in enhancing color prediction accuracy in printing processes. Future research could further explore the transition from 3-channel LAB to 4-channel CMYK data for more comprehensive color mapping.

## Tools and Technologies Used
- PyCharm
- Python
- Color Science Library
- Scikit-learn (sklearn)
- Additional Libraries: NumPy, Pandas, Matplotlib

## Challenges and Solutions
Faced challenges such as limited dataset size and model selection, addressed by experimenting with various settings and models.

## How to Run
- Clone the repository.
- Install the required dependencies: `pip install -r requirements.txt`
- Run the main script: `python Main.py`

## License
This project is licensed under the MIT License - see the LICENSE.md file for details.

## Acknowledgments
Thanks to all the datasets providers and the open-source community for making this research possible.

