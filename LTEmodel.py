import pandas as pd
import numpy as np
import sqlite3
import astropy.constants.codata2014 as const  # h,k_B,c # SI units
from astropy import units as u
import matplotlib.pyplot as plt
import time
import csv
import os
import scipy


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

    # number of spectra to be simulated
    nspec = 10000





    # model parameters in defined ranges
    ntot = 10**np.random.uniform(16, 18, nspec)
    tex = np.random.uniform(10, 400, nspec)
    fwhm = np.random.uniform(1, 10, nspec)
    vlsr = np.random.uniform(-130, 130, nspec)
    size = np.random.uniform(0.2, 2.5, nspec)

    inputs = {'Column density': ntot, 'Excitation temperature': tex, 'FWHM': fwhm, 'Velocity': vlsr, 'Size': size}
    input_data = pd.DataFrame(inputs)
    input_data.take(np.random.permutation(input_data.shape[0]), axis=0)

    # freqmin = 220235
    # freqmax = 220800
    # calculate the minimum frequency of the simulated spectra in MHz from the vlsr velocity
    freqmin = 238858 #238600
    freqmax = 239215 #239180
    length = (freqmax - freqmin) * 10
    intensities = np.zeros(shape=(nspec, length))
    # print(intensities.shape)
    # print(length)

    for i in range(nspec):
        ntot = input_data.iloc[i, 0]
        tex = input_data.iloc[i, 1]
        fwhm = input_data.iloc[i, 2]
        vlsr = input_data.iloc[i, 3]
        size = input_data.iloc[i, 4]


        cpt1 = Component(db, [Species(137, ntot=ntot, tex=tex, fwhm=fwhm)],
                         isInteracting=False, vlsr=vlsr, size=size)
        cpt_list = [cpt1]

        model = ModelSpectrum(cpt_list, fmhz_min=freqmin, fmhz_max=freqmax,
                          eup_max=1000., telescope='alma_400m',
                          tc=tc, tcmb=tcmb, dfmhz=0.01)


        # print("freq, cpt1", [tr.f_trans_mhz for tr in model.get_transition_list(cpt1)])
        # print("tau0, cpt1", [tr.tau0 for tr in model.get_transition_list(cpt1)])
        # print("tau0, cpt2", [tr.tau0 for tr in model.get_transition_list(cpt2)])

        model.compute_model()
        intens = model.intensities
        intensities[i, :] = intens[:length]
        freq = model.frequencies * 1.e-3

        data = [freq, intens]


        file_path = '../spectra/simulated_data1/'

        if not os.path.exists(file_path):
            os.makedirs(file_path)
        file_name = 'model_parameters_data' + str(i) + '.txt'

        with open(file_path + file_name, 'w') as f:
            f.write('-----MODEL PARAMETERS-----\n')
            f.write('tcmb = ' + str(tcmb) + ' K' + '\n')
            f.write('ntot = ' + str(input_data.iloc[i, 0]) + ' cm-2' '\n')
            f.write('tex = ' + str(input_data.iloc[i, 1]) + ' K' + '\n')
            f.write('fwhm = ' + str(input_data.iloc[i, 2]) + ' km/s' + '\n')
            f.write('vlsr = ' + str(input_data.iloc[i, 3]) + ' km/s' + '\n')
            f.write('size = ' + str(input_data.iloc[i, 4]) + ' arsec' + '\n')
            f.write('\n')
            f.write('-----MODEL DATA-----\n')
            f.write('frequency (GHz), intensity (K)\n')

            for x in zip(*data):
                f.write(str(x[0]) + ' ' + str(x[1]) + '\n')
            f.close()

        # plot the spectrum using subplots with figsize=(10, 10)
        # fig, ax = plt.subplots(figsize=(10, 10))
        # ax.plot(freq, intens, color='black')
        # # ax.set_ylim(0, 1.1 * np.max(intens))
        # ax.set_xlabel('Frequency [GHz]')
        # ax.set_ylabel('Intensity [K]')
        # title = ''
        # for cpt in cpt_list:
        #     title += '\n' 'size: ' + str(cpt.size) + ' arsec, ''temperature: ' + str(cpt.tex) + ' K, ' 'velocity: ' + str(cpt.vlsr) + ' km/s'
        #     for sp in cpt.species_list:
        #         title += '\n' + 'total column density: ' + str(sp.ntot) + ' cm-2, ''fwhm: ' + str(sp.fwhm) + ' km/s, ' 'temperature: ' + str(sp.tex) + ' K '
        # ax.set_title(title)
        # plt.show()
        # output_dir = "/Users/user/Dropbox/Manchester/MSc Astronomy & Astrophysics/MSc Research/spectrum/experiment_1/"
        # fig.savefig(output_dir + 'spectrum_' + str(i) + '.png')

    # for icpt, cpt in enumerate(cpt_list):
    #     print('Opacities for component {}'.format(str(icpt + 1)))
    #     for itran, tran in enumerate(model.get_transition_list(cpt)):
    #         print('{} MHz : {}'.format(str(tran.f_trans_mhz), str(tran.tau0)))
    #         ax.plot(np.array([tran.f_trans_mhz, tran.f_trans_mhz]) - cpt.vlsr*1.e3*tran.f_trans_mhz/const.c.value,
    #                 [-np.max(model.intensities)/20, 0],
    #                 color=plot_colors[icpt % len(plot_colors)],
    #                 linestyle=plot_linestyles[icpt % len(plot_linestyles)])

