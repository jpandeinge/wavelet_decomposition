## Discrete Wavelet Decomposition (DWT)

The wavelet decomposition for species of methyl cyanide ($CH_{3}CN$) is implemented simply in this case.

1. We first generate synthetic data using the `LTEmodel.py` script. As a result, the data is saved in the "data/synthetic" folder. After the data has been loaded, the discrete wavelet transform is performed in the notebook "wavelet decomposition.ipynb"

2. In addition, we include our machine learning models using the approximation coefficients of the decomposed signal. By employing the discrete wavelet transform decomposition method with the "Daubechies of order 1" as the mother wavelet at level 6, we are able to forecast the signal's properties, such as the gradient's size, size, and velocity, as well as its column density and excitation temperature.

3. The predicted parameters are then recorded in a csv file in the "data/synthetic/generated_files" folder. Additionally, we used 'LTEmodel reconstruction.py' to rebuild the signal from the predicted parameters.

4. Next, we compare the initial synthesised spectra with the reconstructed spectra produced from the predicted parameters by matching the spectra that were utilised in the assessment of our machine learning models.

5. Observational data from the same species gathered by the Atacama Large Millimeter Array (ALMA) telescope are then used to assess the best model. In order to match the frequency range of the observational data, we kept the frequency range of the data at `238.91 GHz - 239.18 GHz` for the purposes of the basic preprocessing of the data, which was done in the notebook `observational_data_predictions.ipynb`. The best model was used to forecast the parameters after the data were interpolated to have the same amount of points as the synthetic data.
