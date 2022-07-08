## Discrete Wavelet Decomposition (DWT)

The wavelet decomposition for species of methyl cyanide ($CH 3CN$) is implemented simply in this case.

1. First, using the script `LTEmodel.py`, we create synthetic data. The data was therefore saved in the folder labelled `data`. The discrete wavelet transform is then carried out in the notebook `wavelet_decompositin_10k.ipynb` after the data has been loaded.

2. In addition, we use the decomposed signal's approximation coefficients as our features for our machine learning models. As a result, we are able to predict the signal's parameters, such as the gradient's size, size, and velocity, as well as its column density and excitation temperature, using the discrete wavelet transform decomposition methon with the `Daubechies of order 1` as the mother wavelet at level 7.

3. The `10K_gen_files` folder is then where the predicted parameters are saved in a csv file. Additionally, we created the reconstructed signal from the predicted parameters using `LTEmodel_reconstruction.py`.

4. Next, by matching the spectra that were used in the evaluation of our machine learning models, we compare the original synthentic spectra with the reconstructed spectra derived from the predicted parameters.

5. The best model is then evaluated using observational data from the same species collected by the Atacama Large Millimeter Array (ALMA) telescope. The fundamental preprocessing of the data is carried out in the notebook `observational_data_predictions.ipynb`, where we kept the frequency range of the data at `238.60 GHz - 239.18 GHz` to match the frequency range of the synthetic data. The data were interpolated to have the same number of points as the synthetic data, and the best model was used to make parameter predictions.
