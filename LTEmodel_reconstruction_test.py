import sys
import pandas as pd
import numpy as np
import sqlite3
import astropy.constants.codata2014 as const  # h,k_B,c # SI units
from astropy import units as u
import matplotlib.pyplot as plt
import time
import csv
import os
from regex import B
import scipy
from pandas._testing import iloc
from torch import le


class SimpleSpectrum:
    def __init__(self, xarray, yarray, xunit='mhz', yunit='K'):
        self.xval = xarray
        self.yval = yarray
        self.xunit = xunit
        self.yunit = yunit


class ModelSpectrum:
    def __init__(self, cpt_list, fmhz_min=115.e3, fmhz_max=116.e3, dfmhz=0.1, eup_min=0.0, eup_max=150.0, aij_min=0.0,
                 telescope='alma_400m', tcmb=2.73, tc=0):
        self.cpt_list = cpt_list
        self.fmhz_min = fmhz_min
        self.fmhz_max = fmhz_max
        self.dfmhz = dfmhz
        self.eup_min = eup_min
        self.eup_max = eup_max
        self.aij_min = aij_min
        self.telescope = telescope
        self.tcmb = tcmb
        self.tc = tc
        self.frequencies = np.arange(self.fmhz_min, self.fmhz_max, self.dfmhz)
        self.intensities = np.zeros_like(self.frequencies)

    def get_transition_list(self, cpt, species=None):
        if species is None:
            self.species_list = cpt.species_list
        else:
            self.species_list = [species]
        self.db = cpt.db
        transition_list = []
        for sp in self.species_list:
            command = "SELECT * FROM transitions WHERE catdir_id = " + str(sp.tag) + " and fMhz < " + str(self.fmhz_max) + \
                      " and fMhz > " + str(self.fmhz_min) + " and eup < " + str(self.eup_max)
            self.db.execute(command)
            all_rows = self.db.fetchall()
            for row in all_rows:
                trans = Transition(self.db, sp, row[1], row[3], row[4], row[6])  # tag, freq, aij, elo_cm, gup
                transition_list.append(trans)
        transition_list.sort(key=lambda x: x.f_trans_mhz)
        return transition_list

    def compute_model(self):
        cpt_list_interacting = [cpt for cpt in self.cpt_list if cpt.isInteracting]
        cpt_list_non_interacting = [cpt for cpt in self.cpt_list if not cpt.isInteracting]

        tbefore = self.tc + jnu(self.frequencies, self.tcmb)

        intensity_all_cpt = 0.

        for k, cpt in enumerate(cpt_list_non_interacting):
            # dilution_factor = tr.mol_size ** 2 / (tr.mol_size**2 + get_beam_size(model.telescope,freq)**2)
            # dilution_factor = (1. - np.cos(tr.mol_size/3600./180.*np.pi)) / ( (1. - np.cos(tr.mol_size/3600./180.*np.pi))
            #                    + (1. - np.cos(get_beam_size(model.telescope,freq)/3600./180.*np.pi)) )
            ff_g = dilution_factor(cpt.size, get_beam_size(model.telescope, self.frequencies))
            # ff_d = dilution_factor(cpt.size, get_beam_size(model.telescope, self.frequencies), geometry='disc')

            # compute the sum of the difference between outgoing temp and incoming temp
            intensity_all_cpt = intensity_all_cpt + cpt.compute_delta_t_comp(self.frequencies,
                                                                             model.get_transition_list(cpt),
                                                                             tbefore, ff_g)

        # add "background"
        intensity_all_cpt += tbefore

        for k, cpt in enumerate(cpt_list_interacting):
            tbefore = intensity_all_cpt  # re-define the "background" intensity
            # ff_d = 1.
            # ff_g = 1.
            ff_g = dilution_factor(cpt.size, get_beam_size(model.telescope, self.frequencies))
            # ff_d = dilution_factor(cpt.size, get_beam_size(model.telescope, self.frequencies), geometry='disc')
            # sum_tau_cpt = cpt.compute_sum_tau(frequencies, model.get_transition_list(cpt))

            intensity_all_cpt = cpt.compute_delta_t_comp(self.frequencies, model.get_transition_list(cpt),
                                                         tbefore, ff_g, interacting=True) + tbefore

            cpt.assign_spectrum(SimpleSpectrum(self.frequencies, intensity_all_cpt))

        self.intensities = intensity_all_cpt - jnu(self.frequencies, self.tcmb)  # Ton - Toff
        return self.intensities





class Component:
    def __init__(self, db, species_list, isInteracting=False, vlsr=0.0, size=3.0, tex=100.):
        # super().__init__()
        self.db = db
        self.species_list = species_list
        self.tag_list = [sp.tag for sp in self.species_list]
        self.isInteracting = isInteracting
        self.vlsr = vlsr  # km/s
        self.size = size  # arcsec
        self.tex = tex  # K

    def get_fwhm(self, transition):
        tag = transition.tag
        return next(sp for sp in self.species_list if sp.tag == tag).fwhm

    def get_tex(self, transition):
        tag = transition.tag
        return next(sp for sp in self.species_list if sp.tag == tag).tex

    def get_ntot(self, transition):
        tag = transition.tag
        return next(sp for sp in self.species_list if sp.tag == tag).ntot

    def compute_sum_tau(self, fmhz, transition_list):
        sum_tau = 0
        for tr in transition_list:
            num = fmhz - tr.f_trans_mhz + kms_to_mhz(self.vlsr, tr.f_trans_mhz)
            den = fwhm_to_sigma(kms_to_mhz(self.get_fwhm(tr), tr.f_trans_mhz))
            sum_tau += tr.tau0 * np.exp(-0.5 * (num / den) ** 2)
        return sum_tau

    def compute_delta_t_comp(self, fmhz, transition_list, intensity_before, filling_factor, old=False,
                             interacting=False, tcmb=2.73):
        """Computes the difference between outgoing and incoming intensity of a component.
        fmhz : float or numpy array
        component : object Component()
        intensity_before : incoming intensity
        filling_factor : beam filling factor
        """
        # old : deltaT = Tc*(exp(-tau_comp)-1) + sum_mol( ff*(Jtex - Jtcmb)*(1.-exp(-tau_mol))
        # new : deltaT = ff * ( sum_mol(Jtex*(1.-exp(-tau_mol))) + tbefore*(exp(-tau_comp)-1.) )
        sum_tau_cpt = self.compute_sum_tau(fmhz, transition_list)
        deltaT = jnu(fmhz, self.tex) * (1. - np.exp(-sum_tau_cpt)) - intensity_before * (1. - np.exp(-sum_tau_cpt))
        deltaT = deltaT * filling_factor

        return deltaT

    def assign_spectrum(self, spec: SimpleSpectrum):
        self.model_spec = spec

class Species:
    def __init__(self, tag, ntot=7.0e14, tex=100, fwhm=1.0):
        # super().__init__(self)
        self.tag = tag
        self.ntot = ntot  # total column density [cm-2]
        self.tex = tex  # excitation temperature [K]
        self.fwhm = fwhm  # line width [km/s]
        # self.mol_size = mol_size # size of the molecular region[arcsec]


class Transition:
    def __init__(self, db, species, f_trans_mhz, aij, elo_cm, gup):
        self.db = db
        self.f_trans_mhz = f_trans_mhz
        self.aij = aij
        self.elo_cm = elo_cm
        self.elo_J = self.elo_cm * const.h.value * const.c.value * 100
        self.eup_J = self.elo_J + self.f_trans_mhz * 1.e6 * const.h.value
        self.gup = gup
        self.eup = (elo_cm + self.f_trans_mhz * 1.e6 / (const.c.value * 100)) * 1.4389  # [K]
        # print(self.eup)
        self.tag = species.tag
        self.tex = species.tex
        self.ntot = species.ntot
        self.fwhm = species.fwhm

        # def calc_tau0(self):
        # opacity at the line center
        qtex = get_partition_function(self.db, self.tag, self.tex)
        # print("qtex = ", qtex)
        self.nup = self.ntot * self.gup / qtex / np.exp(self.eup_J / const.k_B.value / self.tex)  # [cm-2]
        self.tau0 = const.c.value ** 3 * self.aij * self.nup * 1.e4 \
                    * (np.exp(const.h.value * self.f_trans_mhz * 1.e6 / const.k_B.value / self.tex) - 1.) \
                    / (4. * np.pi * (self.f_trans_mhz * 1.e6) ** 3 * self.fwhm * 1.e3 * np.sqrt(np.pi / np.log(2.)))
        # return tau0


def frange(start, stop, step):
    i = start
    while i < stop:
        yield i
        i += step


def find_nearest(array, value):
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def find_nearest_id(array, value):
    return (np.abs(array - value)).argmin()


def find_nearest_trans(trans_list, value):
    f_trans_list = []
    for tr in trans_list:
        f_trans_list.append(tr.f_trans_mhz)
    idx = (np.abs(np.array(f_trans_list) - value)).argmin()
    return trans_list[idx]


def get_partition_function(db, tag, temp):
    tref = []
    qlog = []
    for row in db.execute("SELECT * FROM cassis_parti_funct WHERE catdir_id = " + str(tag)):
        tref.append(row[1])
        qlog.append(np.power(10., row[2]))
    tref.sort()
    qlog.sort()
    return np.interp(temp, tref, qlog)
    # return np.power(10., qlog[find_nearest_id(np.array(tref),temp)])


def fwhm_to_sigma(value, reverse=False):
    if reverse:
        return value * (2. * np.sqrt(2. * np.log(2.)))
    else:
        return value / (2. * np.sqrt(2. * np.log(2.)))


def kms_to_mhz(value, fref_mhz, reverse=False):
    if reverse:
        return (value / fref_mhz) * const.c.value * 1.e-3
    else:
        return value * 1.e3 * fref_mhz / const.c.value


def jnu(fmhz, temp: float):
    fmhz_arr = np.array(fmhz) if type(fmhz) == list else fmhz
    res = (const.h.value * fmhz_arr * 1.e6 / const.k_B.value) / \
          (np.exp(const.h.value * fmhz_arr * 1.e6 / (const.k_B.value * temp)) - 1.)
    return list(res) if type(fmhz) == list else res


def get_beam_size(tel, freq_mhz):
    tel_dic = {'iram': 30.,
               'apex': 12.,
               'jcmt': 15.,
               'alma_400m': 400.}
    return (1.22 * const.c.value / (freq_mhz * 1.e6)) / tel_dic[tel] * 3600. * 180. / np.pi


def dilution_factor(source_size, beam_size, geometry='gaussian'):
    if geometry == 'disc':
        return 1. - np.exp(-np.log(2.) * (source_size / beam_size) ** 2)
    else:
        return source_size ** 2 / (source_size ** 2 + beam_size ** 2)


if __name__ == '__main__':

    conn = sqlite3.connect('/Users/user/CASSIS/database/cassis20210603.db')
    db = conn.cursor()

    # tc = 1.  # 0.16
    tc = 0.
    tcmb = 2.73



    # open a csv file with the parameters of the spectra  the first row is the header
    PARAMETERS_OUTPUT = 'data/synthetic/generated_files/'
    

    rf_params_df = pd.read_csv(PARAMETERS_OUTPUT + 'predicted_parameters_rf.csv')
    xgb_params_df = pd.read_csv(PARAMETERS_OUTPUT + 'predicted_parameters_xgb.csv')
    xgb_tuned_params_df = pd.read_csv(PARAMETERS_OUTPUT + 'predicted_parameters_xgb_tuned.csv')
    # nn_model_params_df = pd.read_csv(PARAMETERS_OUTPUT + 'predicted_parameters_nn.csv')


    # get the number of spectra to be generated from all the dataframe above
    n_rf = len(rf_params_df)
    n_xgb = len(xgb_params_df)
    n_xgb_tuned = len(xgb_tuned_params_df)
    # n_nn = len(nn_model_params_df)

    # calculate the minimum frequency of the simulated spectra in MHz from the vlsr velocity
    freqmin = 238600 #238600
    freqmax = 239180 #239180
    length = (freqmax - freqmin) * 10

    model_inten_length = [n_rf, n_xgb, n_xgb_tuned] #, n_nn]
    # calculate the intenity to be simulated for each of models
    for model_id in model_inten_length:
        if model_id == n_rf:
            intensities = np.zeros(shape=(model_inten_length[0], length))
        elif model_id == n_xgb:
            intensities = np.zeros(shape=(model_inten_length[1], length))
        elif model_id == n_xgb_tuned:
            intensities = np.zeros(shape=(model_inten_length[2], length))
        # elif model_id == n_nn:
        #     intensities = np.zeros(shape=model_inten_length[3], length=leng)
        else:
            print('Error: model_id not found')
            break


    # intensities = np.zeros(shape=(n_rf, length))
    
    # nn_intensities = np.zeros(shape=(n_nn, length))



    n_spectra = [n_rf, n_xgb, n_xgb_tuned]

    
    for i in range(n_rf):
        ntot = rf_params_df['ntot'].values[i]
        tex =  rf_params_df['tex'].values[i]
        fwhm = rf_params_df['fwhm'].values[i]
        vlsr = rf_params_df['vlsr'].values[i]
        size = rf_params_df['size'].values[i]

    
    for i in range(n_xgb):
        ntot = xgb_params_df['ntot'].values[i]
        tex = xgb_params_df['tex'].values[i]
        fwhm = xgb_params_df['fwhm'].values[i]
        vlsr = xgb_params_df['vlsr'].values[i]
        size = xgb_params_df['size'].values[i]
    
    
    for i in range(n_xgb_tuned):
        ntot = xgb_tuned_params_df['ntot'].values[i]
        tex = xgb_tuned_params_df['tex'].values[i]
        fwhm = xgb_tuned_params_df['fwhm'].values[i]
        vlsr = xgb_tuned_params_df['vlsr'].values[i]
        size = xgb_tuned_params_df['size'].values[i]
    
    
        # get the parameters for each model
    

        cpt1 = Component(db, [Species(137, ntot=ntot, tex=tex, fwhm=fwhm)],
                         isInteracting=False, vlsr=vlsr, size=size)
        cpt_list = [cpt1]

        model = ModelSpectrum(cpt_list, fmhz_min=freqmin, fmhz_max=freqmax,
                          eup_max=1000., telescope='alma_400m',
                          tc=tc, tcmb=tcmb, dfmhz=0.01)


        model.compute_model()
        rf_intens, xgb_intens, xgb_tuned_intens = [] , [] , []
        # get the intesities for each of the models depending on the model_id
        if model_id == n_rf:
            rf_intens = model.intensities
        elif model_id == n_xgb:
            xgb_intens = model.intensities
        elif model_id == n_xgb_tuned:
            xgb_tuned_intens = model.intensities
        # elif model_id == n_nn:
        #     nn_intens = model.intensities
        else:
            print('Error: model_id not found')
            break
       
        # nn_intens = model.intensities
        
        # get the intesities for each of the model
        # intensities[model_id] = model.intensities
       
    
        freq = model.frequencies * 1.e-3


        rf_data = [freq, rf_intens]
        xgb_data = [freq, xgb_intens]
        xgb_tuned_data = [freq, xgb_tuned_intens]
    
        # data = [freq, intens]
        RF_FILE_PATH = 'data/synthetic/spectra/reconstructed/rf/'
        XGB_FILE_PATH = 'data/synthetic/spectra/reconstructed/xgb/'
        XGB_TUNED_FILE_PATH = 'data/synthetic/spectra/reconstructed/xgb_tuned/'
        NN_FILE_PATH = 'data/synthetic/spectra/reconstructed/nn/'

        for path in [RF_FILE_PATH, XGB_FILE_PATH, XGB_TUNED_FILE_PATH]:
            if not os.path.exists(path):
                os.makedirs(path)

        # file_name = 'recon_param_data_plot' + str(i) + '.png'
        text_file_name = 'recon_param_data' + str(i) + '.txt'
    
        model_paths = [RF_FILE_PATH, XGB_FILE_PATH, XGB_TUNED_FILE_PATH]
        # create a  list of all the dataframe of all the models
        model_dataframes = [rf_params_df, xgb_params_df, xgb_tuned_params_df]        

        # loop over all the models and save the spectra to the corresponding path using the appropriate dataframe
        for path, dataframe in zip(model_paths, model_dataframes):
            if path == RF_FILE_PATH:
                with open(path + text_file_name, 'w') as f:
                    f.write('-----MODEL PARAMETERS-----\n')
                    f.write('tcmb = ' + str(tcmb) + ' K' + '\n')
                    f.write('ntot = ' + str(model_dataframes[0]['ntot'].values[i]) + ' cm-2' '\n')
                    f.write('tex = ' + str( model_dataframes[0]['tex'].values[i]) + ' K' + '\n')
                    f.write('fwhm = ' + str(model_dataframes[0]['fwhm'].values[i]) + ' km/s' + '\n')
                    f.write('vlsr = ' + str(model_dataframes[0]['vlsr'].values[i]) + ' km/s' + '\n')
                    f.write('size = ' + str(model_dataframes[0]['size'].values[i]) + ' arsec' + '\n')
                    f.write('\n')
                    f.write('-----MODEL DATA-----\n')
                    f.write('RestFreq(GHz) T(K)\n')
    
                    for x in zip(*rf_data):
                        f.write(str(x[0]) + ' ' + str(x[1]) + '\n')
                    f.close()

            elif path == XGB_FILE_PATH:
                with open(path + text_file_name, 'w') as f:
                    f.write('-----MODEL PARAMETERS-----\n')
                    f.write('tcmb = ' + str(tcmb) + ' K' + '\n')
                    f.write('ntot = ' + str(model_dataframes[1]['ntot'].values[i]) + ' cm-2' '\n')
                    f.write('tex = ' + str( model_dataframes[1]['tex'].values[i]) + ' K' + '\n')
                    f.write('fwhm = ' + str(model_dataframes[1]['fwhm'].values[i]) + ' km/s' + '\n')
                    f.write('vlsr = ' + str(model_dataframes[1]['vlsr'].values[i]) + ' km/s' + '\n')
                    f.write('size = ' + str(model_dataframes[1]['size'].values[i]) + ' arsec' + '\n')
                    f.write('\n')
                    f.write('-----MODEL DATA-----\n')
                    f.write('RestFreq(GHz) T(K)\n')
    
                    for x in zip(*xgb_data):
                        f.write(str(x[0]) + ' ' + str(x[1]) + '\n')
                    f.close()

            elif path == XGB_TUNED_FILE_PATH:
                with open(path + text_file_name, 'w') as f:
                    f.write('-----MODEL PARAMETERS-----\n')
                    f.write('tcmb = ' + str(tcmb) + ' K' + '\n')
                    f.write('ntot = ' + str(model_dataframes[2]['ntot'].values[i]) + ' cm-2' '\n')
                    f.write('tex = ' + str( model_dataframes[2]['tex'].values[i]) + ' K' + '\n')
                    f.write('fwhm = ' + str(model_dataframes[2]['fwhm'].values[i]) + ' km/s' + '\n')
                    f.write('vlsr = ' + str(model_dataframes[2]['vlsr'].values[i]) + ' km/s' + '\n')
                    f.write('size = ' + str(model_dataframes[2]['size'].values[i]) + ' arsec' + '\n')
                    f.write('\n')
                    f.write('-----MODEL DATA-----\n')
                    f.write('RestFreq(GHz) T(K)\n')
    
                    for x in zip(*xgb_tuned_data):
                        f.write(str(x[0]) + ' ' + str(x[1]) + '\n')
                    f.close()
            else:
                print('Error: model path not found')
                sys.exit()

    # for icpt, cpt in enumerate(cpt_list):
    #     print('Opacities for component {}'.format(str(icpt + 1)))
    #     for itran, tran in enumerate(model.get_transition_list(cpt)):
    #         print('{} MHz : {}'.format(str(tran.f_trans_mhz), str(tran.tau0)))
    #         ax.plot(np.array([tran.f_trans_mhz, tran.f_trans_mhz]) - cpt.vlsr*1.e3*tran.f_trans_mhz/const.c.value,
    #                 [-np.max(model.intensities)/20, 0],
    #                 color=plot_colors[icpt % len(plot_colors)],
    #                 linestyle=plot_linestyles[icpt % len(plot_linestyles)])

